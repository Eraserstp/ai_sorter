from __future__ import annotations

import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # noqa: E402

from .database import Database
from .file_utils import iter_top_level_files, move_file, trash_file
from .models import AppSettings, Destination, SortDecision
from .ollama import OllamaClient, OllamaError
from .processor import SortProcessor


class TextList(Gtk.Box):
    def __init__(self, title: str) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.append(Gtk.Label(label=title, xalign=0))
        self.view = Gtk.TextView(monospace=True)
        self.view.set_vexpand(True)
        scroller = Gtk.ScrolledWindow(min_content_height=120)
        scroller.add_css_class("frame")
        scroller.set_child(self.view)
        self.append(scroller)

    def get_lines(self) -> list[str]:
        buffer = self.view.get_buffer()
        text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        return [line.strip() for line in text.splitlines() if line.strip()]

    def set_lines(self, lines: list[str]) -> None:
        self.view.get_buffer().set_text("\n".join(lines))


class AiSorterWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app, title="AI Sorter")
        self.set_default_size(1100, 760)
        self.db = Database()
        self.processor: SortProcessor | None = None
        self.decisions: list[SortDecision] = []
        self.rows: dict[Path, tuple[SortDecision, Gtk.ListBoxRow, Gtk.CheckButton, Gtk.CheckButton, Gtk.CheckButton]] = {}

        self.notebook = Gtk.Notebook()
        self.set_child(self.notebook)
        self._build_settings_tab()
        self._build_destinations_tab()
        self._build_process_tab()
        self.load_all()

    def _build_settings_tab(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        self.source_list = TextList("Каталоги разбора (по одному пути на строку)")
        self.exclusion_list = TextList("Исключенные файлы (полный путь или имя файла, по одному на строку)")
        box.append(self.source_list)
        box.append(self.exclusion_list)

        grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        self.ollama_url = Gtk.Entry()
        self.sorter_model = Gtk.ComboBoxText()
        self.vision_model = Gtk.ComboBoxText()
        refresh = Gtk.Button(label="Получить модели")
        refresh.connect("clicked", self.on_refresh_models)
        save = Gtk.Button(label="Сохранить настройки")
        save.connect("clicked", self.on_save_settings)
        for row, (label, widget) in enumerate((("Ollama URL", self.ollama_url), ("Общая модель", self.sorter_model), ("Мультимодальная модель", self.vision_model))):
            grid.attach(Gtk.Label(label=label, xalign=0), 0, row, 1, 1)
            grid.attach(widget, 1, row, 1, 1)
        grid.attach(refresh, 0, 3, 1, 1)
        grid.attach(save, 1, 3, 1, 1)
        box.append(grid)
        self.notebook.append_page(box, Gtk.Label(label="Настройки"))

    def _build_destinations_tab(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        self.dest_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.dest_list.connect("row-selected", self.on_destination_selected)
        scroller = Gtk.ScrolledWindow(min_content_height=180, vexpand=True)
        scroller.set_child(self.dest_list)
        box.append(scroller)

        grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        self.dest_id: int | None = None
        self.dest_name = Gtk.Entry()
        self.dest_path = Gtk.Entry()
        self.dest_positive = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.dest_negative = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        grid.attach(Gtk.Label(label="Имя", xalign=0), 0, 0, 1, 1)
        grid.attach(self.dest_name, 1, 0, 1, 1)
        path_box = Gtk.Box(spacing=6)
        browse_path = Gtk.Button(label="Выбрать…")
        browse_path.connect("clicked", self.on_choose_destination_path)
        path_box.append(self.dest_path)
        path_box.append(browse_path)
        self.dest_path.set_hexpand(True)
        grid.attach(Gtk.Label(label="Путь", xalign=0), 0, 1, 1, 1)
        grid.attach(path_box, 1, 1, 1, 1)
        grid.attach(Gtk.Label(label="Позитивный промпт", xalign=0), 0, 2, 1, 1)
        grid.attach(self._wrap_textview(self.dest_positive, 120), 1, 2, 1, 1)
        grid.attach(Gtk.Label(label="Негативный промпт", xalign=0), 0, 3, 1, 1)
        grid.attach(self._wrap_textview(self.dest_negative, 120), 1, 3, 1, 1)
        box.append(grid)
        buttons = Gtk.Box(spacing=8)
        for label, handler in (("Новый", self.on_new_destination), ("Сохранить", self.on_save_destination), ("Удалить", self.on_delete_destination)):
            button = Gtk.Button(label=label)
            button.connect("clicked", handler)
            buttons.append(button)
        box.append(buttons)
        self.notebook.append_page(box, Gtk.Label(label="Назначения"))

    def _build_process_tab(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        self.summary = Gtk.Label(xalign=0, wrap=True)
        box.append(self.summary)
        controls = Gtk.Box(spacing=8)
        start = Gtk.Button(label="Начать обработку")
        start.connect("clicked", self.on_start_processing)
        stop = Gtk.Button(label="Остановить")
        stop.connect("clicked", self.on_stop_processing)
        apply = Gtk.Button(label="Обработать выбранное")
        apply.connect("clicked", self.on_apply_actions)
        controls.append(start); controls.append(stop); controls.append(apply)
        box.append(controls)
        self.progress = Gtk.TextView(editable=False, monospace=True)
        progress_scroller = Gtk.ScrolledWindow(min_content_height=90)
        progress_scroller.add_css_class("frame")
        progress_scroller.set_child(self.progress)
        box.append(progress_scroller)
        self.result_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.result_list.connect("row-activated", self.on_result_activated)
        result_scroller = Gtk.ScrolledWindow(vexpand=True)
        result_scroller.set_child(self.result_list)
        box.append(result_scroller)
        self.notebook.append_page(box, Gtk.Label(label="Процесс"))

    def _wrap_textview(self, view: Gtk.TextView, height: int) -> Gtk.ScrolledWindow:
        scroller = Gtk.ScrolledWindow(min_content_height=height)
        scroller.add_css_class("frame")
        view.set_left_margin(6)
        view.set_right_margin(6)
        view.set_top_margin(6)
        view.set_bottom_margin(6)
        scroller.set_child(view)
        return scroller

    def on_choose_destination_path(self, _button: Gtk.Button) -> None:
        chooser = Gtk.FileChooserNative.new(
            "Выберите каталог назначения",
            self,
            Gtk.FileChooserAction.SELECT_FOLDER,
            "Выбрать",
            "Отмена",
        )
        current = self.dest_path.get_text().strip()
        if current:
            folder = Path(current).expanduser()
            if folder.exists():
                chooser.set_current_folder(Gio.File.new_for_path(str(folder)))
        chooser.connect("response", self.on_destination_path_chosen)
        chooser.show()

    def on_destination_path_chosen(self, chooser: Gtk.FileChooserNative, response: int) -> None:
        if response == Gtk.ResponseType.ACCEPT:
            file = chooser.get_file()
            if file:
                self.dest_path.set_text(file.get_path() or "")
        chooser.destroy()

    def _text(self, view: Gtk.TextView) -> str:
        buffer = view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)

    def _set_text(self, view: Gtk.TextView, text: str) -> None:
        view.get_buffer().set_text(text)

    def load_all(self) -> None:
        settings = self.db.get_settings()
        self.ollama_url.set_text(settings.ollama_url)
        self.source_list.set_lines([item.path for item in self.db.list_sources()])
        self.exclusion_list.set_lines([item.path for item in self.db.list_exclusions()])
        self.refresh_destinations()
        self.refresh_summary()
        self.set_combo_text(self.sorter_model, settings.sorter_model)
        self.set_combo_text(self.vision_model, settings.vision_model)

    def set_combo_text(self, combo: Gtk.ComboBoxText, text: str) -> None:
        combo.remove_all()
        if text:
            combo.append_text(text)
            combo.set_active(0)

    def on_refresh_models(self, _button: Gtk.Button) -> None:
        try:
            models = OllamaClient(self.ollama_url.get_text()).list_models()
        except OllamaError as exc:
            self.log(str(exc)); return
        current_sorter = self.sorter_model.get_active_text()
        current_vision = self.vision_model.get_active_text()
        self.sorter_model.remove_all(); self.vision_model.remove_all()
        for model in models:
            self.sorter_model.append_text(model); self.vision_model.append_text(model)
        for combo, current in ((self.sorter_model, current_sorter), (self.vision_model, current_vision)):
            combo.set_active(max(0, models.index(current) if current in models else 0) if models else -1)

    def on_save_settings(self, _button: Gtk.Button) -> None:
        self.db.save_settings(AppSettings(self.ollama_url.get_text(), self.sorter_model.get_active_text() or "", self.vision_model.get_active_text() or ""))
        self.db.replace_sources(self.source_list.get_lines())
        self.db.replace_exclusions(self.exclusion_list.get_lines())
        self.refresh_summary(); self.log("Настройки сохранены")

    def refresh_destinations(self) -> None:
        while row := self.dest_list.get_row_at_index(0):
            self.dest_list.remove(row)
        for dest in self.db.list_destinations():
            row = Gtk.ListBoxRow()
            row.destination = dest  # type: ignore[attr-defined]
            row.set_child(Gtk.Label(label=f"{dest.name} → {dest.path}", xalign=0))
            self.dest_list.append(row)

    def on_destination_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if not row:
            return
        dest: Destination = row.destination  # type: ignore[attr-defined]
        self.dest_id = dest.id
        self.dest_name.set_text(dest.name); self.dest_path.set_text(dest.path)
        self._set_text(self.dest_positive, dest.positive_prompt); self._set_text(self.dest_negative, dest.negative_prompt)

    def on_new_destination(self, _button: Gtk.Button) -> None:
        self.dest_id = None; self.dest_name.set_text(""); self.dest_path.set_text("")
        self._set_text(self.dest_positive, ""); self._set_text(self.dest_negative, "")

    def on_save_destination(self, _button: Gtk.Button) -> None:
        self.db.upsert_destination(Destination(self.dest_id, self.dest_name.get_text(), self.dest_path.get_text(), self._text(self.dest_positive), self._text(self.dest_negative)))
        self.refresh_destinations(); self.refresh_summary(); self.log("Назначение сохранено")

    def on_delete_destination(self, _button: Gtk.Button) -> None:
        if self.dest_id is not None:
            self.db.delete_destination(self.dest_id)
            self.on_new_destination(_button); self.refresh_destinations(); self.refresh_summary()

    def refresh_summary(self) -> None:
        sources = "\n".join(f" • {item.path}" for item in self.db.list_sources()) or " • не заданы"
        destinations = "\n".join(f" • {item.name}: {item.path}" for item in self.db.list_destinations()) or " • не заданы"
        self.summary.set_text(f"Каталоги разбора:\n{sources}\n\nНазначения:\n{destinations}")

    def on_start_processing(self, _button: Gtk.Button) -> None:
        self.on_save_settings(_button)
        files = iter_top_level_files([item.path for item in self.db.list_sources()], {item.path for item in self.db.list_exclusions()})
        self.processor = SortProcessor(self.db, lambda message: GLib.idle_add(self.log, message))
        threading.Thread(target=self._process_worker, args=(files,), daemon=True).start()

    def _process_worker(self, files: list[Path]) -> None:
        try:
            assert self.processor is not None
            self.processor.prepare_media_analysis(files)
            decisions = self.processor.classify_files(files)
            GLib.idle_add(self.show_decisions, decisions)
        except Exception as exc:  # noqa: BLE001 - surface background failures to UI
            GLib.idle_add(self.log, f"Ошибка обработки: {exc}")

    def on_stop_processing(self, _button: Gtk.Button) -> None:
        if self.processor:
            self.processor.cancel(); self.log("Запрошена остановка")

    def show_decisions(self, decisions: list[SortDecision]) -> None:
        self.decisions = decisions; self.rows.clear()
        while row := self.result_list.get_row_at_index(0):
            self.result_list.remove(row)
        for decision in decisions:
            row = Gtk.ListBoxRow()
            grid = Gtk.Grid(column_spacing=10, row_spacing=4, margin_top=6, margin_bottom=6, margin_start=6, margin_end=6)
            grid.attach(Gtk.Label(label=decision.file_path.name, xalign=0), 0, 0, 1, 1)
            destination_label = decision.destination_name if decision.action != "delete" else "DELETE / удалить"
            grid.attach(Gtk.Label(label=destination_label, xalign=0), 1, 0, 1, 1)
            confidence = Gtk.Label(label=f"{decision.confidence}%", xalign=0)
            confidence.add_css_class("success" if decision.confidence >= 50 else "error")
            grid.attach(confidence, 2, 0, 1, 1)
            grid.attach(Gtk.Label(label=decision.reason, xalign=0, wrap=True), 0, 1, 3, 1)
            agree = Gtk.CheckButton(label="Согласен")
            disagree = Gtk.CheckButton(label="Несогласен")
            delete = Gtk.CheckButton(label="Удалить")
            disagree.set_active(True)
            for button in (agree, disagree, delete):
                button.connect("toggled", self.on_action_toggled, decision.file_path, (agree, disagree, delete))
                grid.attach(button, 3 + (agree, disagree, delete).index(button), 0, 1, 1)
            row.set_child(grid); row.decision = decision  # type: ignore[attr-defined]
            self.rows[decision.file_path] = (decision, row, agree, disagree, delete)
            self.result_list.append(row)
        self.log(f"Готово решений: {len(decisions)}")

    def on_action_toggled(self, button: Gtk.CheckButton, path: Path, buttons: tuple[Gtk.CheckButton, Gtk.CheckButton, Gtk.CheckButton]) -> None:
        if not button.get_active():
            return
        for other in buttons:
            if other is not button:
                other.set_active(False)
        state = button.get_label()
        event = Gtk.get_current_event()
        if event and event.get_modifier_state() & Gdk.ModifierType.SHIFT_MASK and state:
            for _decision, _row, agree, disagree, delete in self.rows.values():
                target = {"Согласен": agree, "Несогласен": disagree, "Удалить": delete}.get(state)
                if target:
                    target.set_active(True)

    def on_apply_actions(self, _button: Gtk.Button) -> None:
        dialog = Gtk.AlertDialog(message="Подтвердить перемещение и удаление выбранных файлов?")
        dialog.set_buttons(["Отмена", "Подтвердить"])
        dialog.choose(self, None, self._apply_actions_confirmed)

    def _apply_actions_confirmed(self, dialog: Gtk.AlertDialog, result: object) -> None:
        if dialog.choose_finish(result) != 1:
            return
        for path, (decision, row, agree, disagree, delete) in list(self.rows.items()):
            try:
                if agree.get_active() and decision.destination_path:
                    move_file(path, Path(decision.destination_path))
                    self.result_list.remove(row)
                    del self.rows[path]
                elif delete.get_active():
                    trash_file(path)
                    self.result_list.remove(row)
                    del self.rows[path]
            except Exception as exc:  # noqa: BLE001
                self.log(f"Не удалось обработать {path.name}: {exc}")
        self.log("Операции завершены. В списке оставлены несогласованные файлы.")

    def on_result_activated(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        decision: SortDecision = row.decision  # type: ignore[attr-defined]
        dialog = ManualCorrectionDialog(self, self.db.list_destinations())
        dialog.connect("response", self.on_manual_correction_response, decision, dialog)
        dialog.present()

    def on_manual_correction_response(self, dialog: Gtk.Dialog, response: int, decision: SortDecision, correction: "ManualCorrectionDialog") -> None:
        if response != Gtk.ResponseType.OK:
            dialog.destroy(); return
        manual = correction.selected_destination()
        if not manual:
            dialog.destroy(); return
        reason = correction.reason()
        dialog.destroy()
        try:
            wrong = next((dest for dest in self.db.list_destinations() if dest.id == decision.destination_id), None)
            positive, negative = SortProcessor(self.db).suggest_prompt_update(decision.file_path, manual, wrong, reason)
            confirm = Gtk.AlertDialog(message=f"Сохранить новые промпты и переобработать?\n\nPositive для {manual.name}:\n{positive}\n\nNegative для {wrong.name if wrong else 'ошибочного назначения'}:\n{negative}")
            confirm.set_buttons(["Нет", "Да"])
            confirm.choose(self, None, lambda d, r: self._save_prompt_update(d, r, manual, wrong, positive, negative))
        except Exception as exc:  # noqa: BLE001
            self.log(f"Не удалось предложить промпты: {exc}")

    def _save_prompt_update(self, dialog: Gtk.AlertDialog, result: object, manual: Destination, wrong: Destination | None, positive: str, negative: str) -> None:
        if dialog.choose_finish(result) != 1:
            return
        self.db.upsert_destination(Destination(manual.id, manual.name, manual.path, positive, manual.negative_prompt))
        if wrong:
            self.db.upsert_destination(Destination(wrong.id, wrong.name, wrong.path, wrong.positive_prompt, negative))
        self.refresh_destinations(); self.on_start_processing(Gtk.Button())

    def log(self, message: str) -> bool:
        buffer = self.progress.get_buffer()
        buffer.insert(buffer.get_end_iter(), message + "\n")
        return False


class ManualCorrectionDialog(Gtk.Dialog):
    def __init__(self, parent: Gtk.Window, destinations: list[Destination]) -> None:
        super().__init__(title="Ручное назначение", transient_for=parent, modal=True)
        self.destinations = destinations
        self.add_button("Отмена", Gtk.ResponseType.CANCEL)
        self.add_button("Сохранить", Gtk.ResponseType.OK)
        box = self.get_content_area()
        self.combo = Gtk.ComboBoxText()
        for dest in destinations:
            self.combo.append(str(dest.id), f"{dest.name} → {dest.path}")
        if destinations:
            self.combo.set_active(0)
        self.reason_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        box.append(Gtk.Label(label="Выберите правильное назначение и опишите причину", xalign=0))
        box.append(self.combo)
        scroller = Gtk.ScrolledWindow(min_content_height=120)
        scroller.add_css_class("frame")
        self.reason_view.set_left_margin(6)
        self.reason_view.set_right_margin(6)
        self.reason_view.set_top_margin(6)
        self.reason_view.set_bottom_margin(6)
        scroller.set_child(self.reason_view)
        box.append(scroller)

    def selected_destination(self) -> Destination | None:
        active = self.combo.get_active_id()
        return next((dest for dest in self.destinations if str(dest.id) == active), None)

    def reason(self) -> str:
        buffer = self.reason_view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)


class AiSorterApplication(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="io.github.ai_sorter")

    def do_activate(self) -> None:
        self.props.active_window.present() if self.props.active_window else AiSorterWindow(self).present()


def main() -> None:
    app = AiSorterApplication()
    raise SystemExit(app.run())


if __name__ == "__main__":
    main()
