import json
from bisect import bisect_left
from bisect import bisect_right
from bisect import insort
from enum import auto
from enum import Enum
from itertools import chain
from operator import add
from operator import sub
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import dearpygui.dearpygui as dpg
import numpy as np
from scipy.interpolate import CubicSpline

from egm_analyzer.models.model import PredictionModel
from egm_analyzer.models.onnx_wrapper import OnnxModelWrapper
from egm_analyzer.pred_processor import Compressor
from egm_analyzer.predictions_postprocess import postprocess_predictions
from egm_analyzer.signal_processor import SignalProcessor
from egm_analyzer.types import Hz
from egm_analyzer.types import Index
from egm_analyzer.types import InferenceMeta
from egm_analyzer.types import InferenceResult
from egm_analyzer.types import Peak
from egm_analyzer.types import Stage


def find_relative_minimum_derivative_index(
    signal_cutout: np.ndarray,
    signal_to_target_frequency: float,
) -> Index:
    indexes = range(len(signal_cutout))
    cs = CubicSpline(indexes, signal_cutout)

    high_res_indexes = np.linspace(
        0,
        len(signal_cutout) - 1,
        int(signal_to_target_frequency * (len(signal_cutout) - 1)),
        endpoint=False,
    )

    min_derivative_x = int(np.argmin(cs(high_res_indexes, 1)))

    return min_derivative_x / signal_to_target_frequency


def by_position(x: Peak) -> Index:
    return x.position


def find_le_index(peaks: Sequence[Peak], index: Index) -> int:
    'Find rightmost value less than or equal to x'

    i = bisect_right(peaks, index, key=by_position)

    if i:
        return max(0, i)

    return 0


def find_ge_index(peaks: Sequence[Peak], x: Index) -> int:
    'Find leftmost item greater than or equal to x'

    i = bisect_left(peaks, x, key=by_position)
    if i != len(peaks):
        return i

    return max(0, len(peaks) - 1)


def find_peak(position: Index, peaks: Sequence[Peak]) -> int:
    'Locate the leftmost value exactly equal to x'
    return bisect_left(peaks, position, key=by_position)

def generate_uuid() -> str:
    return str(uuid4())


class Direction(Enum):
    LEFT = auto()
    RIGHT = auto()


class Main(object):
    _SIGNAL_FREQUENCY: Hz = 5000

    # GUI tags
    # Main Window Layout
    _main_window_tag = generate_uuid()
    _left_column_tag = generate_uuid()

    # Signal Processing Layout
    _signal_processor_window_tag = generate_uuid()
    _model_label_tag = generate_uuid()
    _gpu_memory_int_field_tag = generate_uuid()
    _intersection_width_tag = generate_uuid()
    _postprocessing_checkbox_tag = generate_uuid()
    _postprocessing_window_width_tag = generate_uuid()
    _target_frequency_tag = generate_uuid()
    _threshold_input_tag = generate_uuid()
    _batch_size_input_tag = generate_uuid()

    # Settings Layout
    _settings_window_tag = generate_uuid()
    _signal_cutout_length_tag = generate_uuid()
    _step_size_tag = generate_uuid()
    _search_segment_width_input_tag = generate_uuid()

    # Plots
    # Top plot (downsampled full signal)
    _top_signal_plot_tag = generate_uuid()
    _top_signal_y_axis_tag = generate_uuid()
    _top_signal_x_axis_tag = generate_uuid()

    # Bottom plot
    _bottom_signal_plot_tag = generate_uuid()
    _bottom_signal_y_axis_tag = generate_uuid()
    _bottom_signal_x_axis_tag = generate_uuid()

    _channel_top_series_tags: dict[int, str] = {}
    _editable_channel_series_tag = generate_uuid()

    _signal_file_dialog_tag = generate_uuid()
    _model_file_dialog_tag = generate_uuid()
    _label_file_dialog_tag = generate_uuid()
    _inference_result_dialog_tag = generate_uuid()

    _editable_channel_input_tag = generate_uuid()
    _next_btn_tag = generate_uuid()
    _prev_btn_tag = generate_uuid()
    _start_input_tag = generate_uuid()
    _update_peaks_btn_tag = generate_uuid()
    _cancel_btn_tag = generate_uuid()

    _font_tag = generate_uuid()

    _channel_checkbox_tags: list[str] = []
    _peaks: list[list[Peak]] = []
    _drag_lines: list[str] = []

    # Edited drag lines
    _edited_drag_lines: dict[str, float] = {}
    _added_drag_lines: list[str] = []
    _deleted_drag_lines: list[str] = []

    # Some placeholders
    _signal: np.ndarray | None = None
    _inference_result_path: Path | None = None
    _model: PredictionModel | None = None
    _signal_path: Path | None = None
    _model_path: Path | None = None

    def __init__(self, *, scale: float = 1.0) -> None:
        self.scale = lambda x: round(x * scale)
        self.sth = lambda x: round(x * self._SIGNAL_FREQUENCY)

        dpg.create_context()

        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=self._space_pressed_callback)

        # Set up font for cyrillic
        font_path = Path(__file__).parent.parent / 'assets/fonts/NotoSans-Regular.ttf'
        with dpg.font_registry():
            with dpg.font(
                font_path.as_posix(),
                self.scale(17),
                tag=self._font_tag,
            ):
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)

        with dpg.file_dialog(
            show=False,
            callback=self._open_signal_file,
            tag=self._signal_file_dialog_tag,
            width=700,
            height=400,
        ):
            dpg.add_file_extension('.npy', color=(0, 255, 0, 255), custom_text='[Numpy]')

        with dpg.file_dialog(
            show=False,
            callback=self._load_model_callback,
            tag=self._model_file_dialog_tag,
            width=700,
            height=400,
        ):
            dpg.add_file_extension('.onnx', color=(0, 255, 0, 255), custom_text='[ONNX]')

        with dpg.file_dialog(
            show=False,
            callback=self._load_inference_result_callback,
            tag=self._inference_result_dialog_tag,
            width=700,
            height=400,
        ):
            dpg.add_file_extension('.json', color=(0, 255, 0, 255), custom_text='[Json]')

        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=self._save_as_label_file_callback,
            tag=self._label_file_dialog_tag,
            width=700,
            height=400,
        )

        with dpg.window(
            label='Настройки',
            show=False,
            tag=self._settings_window_tag,
        ):
            dpg.add_input_float(
                label='Ширина сигнала, с',
                tag=self._signal_cutout_length_tag,
                default_value=2,
                min_clamped=True,
                min_value=0,
                width=self.scale(100),
                format='%.2f',
                callback=self._update_on_signal_cutout_change_callback,
            )
            dpg.add_input_float(
                label='Длина шага, с',
                tag=self._step_size_tag,
                default_value=1.5,
                min_clamped=True,
                min_value=0,
                width=self.scale(100),
                format='%.2f',
            )
            dpg.add_input_int(
                label='Длина окна поиска производной',
                tag=self._search_segment_width_input_tag,
                default_value=50,
                min_value=5,
                width=self.scale(100),
            )

        with dpg.window(
            show=False,
            tag=self._signal_processor_window_tag,
            width=self.scale(400),
            height=self.scale(300),
            no_collapse=True,
            no_resize=True,
        ):
            # Inference settings
            dpg.add_input_int(
                label='Память ГПУ, Гб',
                tag=self._gpu_memory_int_field_tag,
                default_value=8,
                width=self.scale(120),
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label='Выбрать модель',
                    callback=lambda: dpg.show_item(self._model_file_dialog_tag),
                    width=self.scale(120),
                )
                dpg.add_text('', tag=self._model_label_tag)

            dpg.add_input_int(
                label='Пересечение, индекс',
                tag=self._intersection_width_tag,
                default_value=1000,
                width=self.scale(120),
            )
            dpg.add_input_int(
                label='Размер батча',
                tag=self._batch_size_input_tag,
                default_value=400,
                width=self.scale(120),
            )
            dpg.add_input_float(
                label='Пороговое значение',
                tag=self._threshold_input_tag,
                default_value=0.6,
                width=self.scale(120),
                max_clamped=True,
                min_clamped=True,
                max_value=1,
                min_value=0,
            )

            # Spline settings
            dpg.add_input_int(
                label='Целевая частота, КГц',
                tag=self._target_frequency_tag,
                default_value=20,
                width=self.scale(120),
            )

            # Postprocessing settings
            dpg.add_checkbox(
                label='Постобработка',
                tag=self._postprocessing_checkbox_tag,
                default_value=True,
            )
            dpg.add_input_int(
                label='Размер окна постобработки, индекс',
                tag=self._postprocessing_window_width_tag,
                default_value=200,
                width=self.scale(120),
            )

            dpg.add_button(label='Пуск', width=-1, callback=self._predict_callback)

        with dpg.window(tag=self._main_window_tag):
            with dpg.menu_bar():
                with dpg.menu(label='Сигнал'):
                    dpg.add_menu_item(
                        label='Открыть',
                        callback=lambda: dpg.show_item(self._signal_file_dialog_tag),
                    )
                    dpg.add_menu_item(
                        label='Обработать',
                        callback=lambda: dpg.show_item(self._signal_processor_window_tag),
                    )
                    dpg.add_menu_item(
                        label='Настройки',
                        callback=lambda: dpg.show_item(self._settings_window_tag),
                    )

                with dpg.menu(label='Разметка'):
                    dpg.add_menu_item(
                        label='Открыть',
                        callback=lambda: dpg.show_item(self._inference_result_dialog_tag),
                    )
                    dpg.add_menu_item(
                        label='Сохранить',
                        callback=self._save_label_file_callback,
                    )
                    dpg.add_menu_item(
                        label='Сохранить как...',
                        callback=lambda: dpg.show_item(self._label_file_dialog_tag),
                    )

            with dpg.group(horizontal=True):
                with dpg.group(width=100):
                    dpg.add_text('Канал:')
                    dpg.add_input_int(
                        default_value=1,
                        width=-1,
                        tag=self._editable_channel_input_tag,
                        callback=self._update_editable_signal_callback,
                        min_clamped=True,
                        min_value=1,
                    )
                    dpg.add_child_window(width=-1, height=-1, tag=self._left_column_tag)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(
                            label='Старт, сек',
                            width=self.scale(120),
                            format='%.4f',
                            min_clamped=True,
                            min_value=0,
                            default_value=0,
                            tag=self._start_input_tag,
                            callback=self._update_plots_on_start_change_callback,
                        )
                        dpg.add_button(
                            label='Назад',
                            tag=self._prev_btn_tag,
                            callback=self._make_step_callback,
                            user_data=Direction.LEFT,
                        )
                        dpg.add_button(
                            label='Далее',
                            tag=self._next_btn_tag,
                            callback=self._make_step_callback,
                            user_data=Direction.RIGHT,
                        )

                    with dpg.plot(
                        width=-1,
                        height=self.scale(400),
                        anti_aliased=True,
                        tag=self._top_signal_plot_tag,
                        tracked=True,
                    ):
                        # optionally create legend
                        dpg.add_plot_legend()

                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label='Время, с',
                            tag=self._top_signal_x_axis_tag,
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis,
                            label='Напряжение',
                            tag=self._top_signal_y_axis_tag,
                        )

                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label='Добавить метку', callback=self._add_peak_callback,
                        )
                        dpg.add_button(
                            label='Удалить все видимые метки',
                            callback=self._delete_peaks_callback,
                        )
                        dpg.add_button(
                            label='Применить',
                            tag=self._update_peaks_btn_tag,
                            callback=self._update_peaks_callback,
                        )
                        dpg.add_button(
                            label='Отмена',
                            tag=self._cancel_btn_tag,
                            show=False,
                            callback=self._cancel_changes_callback,
                        )

                    with dpg.plot(
                        width=-1,
                        height=-1,
                        anti_aliased=True,
                        tag=self._bottom_signal_plot_tag,
                    ):
                        # optionally create legend
                        dpg.add_plot_legend()

                        # REQUIRED: create x and y axes
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label='Время, с',
                            tag=self._bottom_signal_x_axis_tag,
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis,
                            label='Напряжение',
                            tag=self._bottom_signal_y_axis_tag,
                        )

                        dpg.add_line_series(
                            [],
                            [],
                            tag=self._editable_channel_series_tag,
                            parent=self._bottom_signal_y_axis_tag,
                        )

        dpg.bind_font(self._font_tag)

    def start_app(self) -> None:
        dpg.create_viewport(
            title='EGM Analyzer',
            width=self.scale(1000),
            height=self.scale(800),
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self._main_window_tag, True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _space_pressed_callback(self) -> None:
        if dpg.is_item_hovered(self._bottom_signal_plot_tag):
            x, *__ = dpg.get_plot_mouse_pos()
            self._add_peak(x)

    def _cancel_changes_callback(self, sender: str) -> None:
        start = self.sth(dpg.get_value(self._start_input_tag))
        length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
        stop = start + length
        channel = dpg.get_value(self._editable_channel_input_tag) - 1

        self._update_drag_lines(start, stop, channel)
        self._cancel_changes()

    def _cancel_changes(self) -> None:
        self._edited_drag_lines = {}
        self._deleted_drag_lines = []
        self._added_drag_lines = []

        dpg.set_item_label(self._update_peaks_btn_tag, 'Применить')
        dpg.hide_item(self._cancel_btn_tag)

    def _add_peak_callback(self) -> None:
        (
            start,
            stop,
        ) = dpg.get_axis_limits(self._bottom_signal_x_axis_tag)
        position = (stop + start) / 2
        self._add_peak(position)

    def _add_peak(self, pos: float) -> None:
        dpg.show_item(self._cancel_btn_tag)
        dpg.set_item_label(self._update_peaks_btn_tag, '*Применить')

        drag_line = generate_uuid()
        dpg.add_drag_line(
            color=(0, 255, 0, 150),
            thickness=3,
            default_value=pos,
            tag=drag_line,
            parent=self._bottom_signal_plot_tag,
            callback=self._update_peaks_on_drag_callback,
            user_data=Peak(
                position=pos * self._SIGNAL_FREQUENCY,
                creation_stage=Stage.HUMAN_EDIT,
                search_segment=None,
            ),
        )
        self._drag_lines.append(drag_line)
        self._added_drag_lines.append(drag_line)

    def _delete_peaks_callback(self) -> None:
        dpg.show_item(self._cancel_btn_tag)
        dpg.set_item_label(self._update_peaks_btn_tag, '*Применить')

        (
            start,
            stop,
        ) = dpg.get_axis_limits(self._bottom_signal_x_axis_tag)

        for drag_line in chain(self._drag_lines, self._edited_drag_lines):
            position = getattr(dpg.get_item_user_data(drag_line), 'position', None)

            # In case if line was edited
            if pos := self._edited_drag_lines.get(drag_line, None):
                position = pos

            if position and start < position / self._SIGNAL_FREQUENCY < stop:
                self._deleted_drag_lines.append(drag_line)
                dpg.hide_item(drag_line)

    def _update_peaks_callback(self, sender: str) -> None:
        dpg.set_item_label(sender, 'Применить')

        channel = dpg.get_value(self._editable_channel_input_tag) - 1

        added = set(self._added_drag_lines)
        edited = set(self._edited_drag_lines.keys())
        deleted = set(self._deleted_drag_lines)

        added_edited = added & edited
        edited_deleted = edited & deleted
        added_deleted = added & deleted

        added_edited_deleted = added & edited & deleted

        to_delete = deleted - added_deleted
        to_edit = edited - added - deleted
        to_add = added - deleted

        for drag_line in to_delete:
            peak: Peak | None = dpg.get_item_user_data(drag_line)

            if not peak:
                continue

            index = find_peak(peak.position, self._peaks[channel])
            del self._peaks[channel][index]

        for drag_line in to_edit:
            peak: Peak | None = dpg.get_item_user_data(drag_line)

            if not peak:
                continue

            pos = peak.position
            index = find_peak(pos, self._peaks[channel])
            new_pos = self._edited_drag_lines.get(drag_line, pos)
            new_pos = self._improve_peak_position(new_pos, channel)
            self._peaks[channel][index] = Peak(
                position=new_pos,
                creation_stage=Stage.HUMAN_EDIT,
                search_segment=None,
            )

        for drag_line in to_add:
            peak: Peak | None = dpg.get_item_user_data(drag_line)

            if not peak:
                continue

            pos = self._edited_drag_lines.get(drag_line, peak.position)
            pos = self._improve_peak_position(pos, channel)

            peak_to_insert = Peak(
                position=pos,
                creation_stage=Stage.HUMAN_EDIT,
                search_segment=None,
            )

            insort(self._peaks[channel], peak_to_insert, key=by_position)

        self._cancel_changes()

        start = self.sth(dpg.get_value(self._start_input_tag))
        length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
        stop = start + length

        self._update_drag_lines(start, stop, channel)

    def _improve_peak_position(self, position: Index, channel: int) -> Index:
        if self._signal is None:
            raise RuntimeError('You need to load signal file')

        width = dpg.get_value(self._search_segment_width_input_tag)
        half_width = width // 2

        start = max(0, round(position - half_width))
        stop = start + width

        if stop > len(self._signal[channel]) - 1:
            stop = len(self._signal[channel]) - 1
            start = stop - width

        relative_index = find_relative_minimum_derivative_index(
            self._signal[channel][start:stop],
            20_000 / self._SIGNAL_FREQUENCY,  # TODO: replace with parameter
        )

        return start + relative_index

    def _load_inference_result_callback(self, sender: str, app_data: dict) -> None:
        try:
            filepath, *__ = app_data['selections'].values()
        except ValueError:
            return

        path = Path(filepath)
        self._inference_result_path = path

        inference_result = InferenceResult(**json.loads(path.read_text()))

        self._peaks = inference_result.peaks

        if self._signal is not None:
            start = self.sth(dpg.get_value(self._start_input_tag))
            length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
            stop = start + length
            channel = dpg.get_value(self._editable_channel_input_tag) - 1

            self._update_editable_signal(start, length, channel)
            self._update_drag_lines(start, stop, channel)



    def _update_on_signal_cutout_change_callback(
        self, sender: str, length_input: float,
    ) -> None:
        self._cancel_changes()

        start = self.sth(dpg.get_value(self._start_input_tag))
        length = self.sth(length_input)
        stop = start + length
        channel = dpg.get_value(self._editable_channel_input_tag) - 1

        self._update_editable_signal(start, stop, channel)
        self._update_drag_lines(start, stop, channel)
        self._update_top_signal(start, stop)

    def _update_drag_lines(self, start: int, stop: int, channel: int) -> None:
        # Cleanup
        for drag_line in self._drag_lines:
            dpg.delete_item(drag_line)

        self._drag_lines = []

        peaks = self._peaks[channel]
        start_index = find_ge_index(peaks, start)
        stop_index = find_le_index(peaks, stop)

        for i in range(start_index, stop_index):
            drag_line_tag = generate_uuid()
            peak = peaks[i]
            dpg.add_drag_line(
                default_value=peak.position / self._SIGNAL_FREQUENCY,
                user_data=peak,
                callback=self._update_peaks_on_drag_callback,
                tag=drag_line_tag,
                parent=self._bottom_signal_plot_tag,
                color=(255, 0, 0, 150),
                thickness=3,
            )
            self._drag_lines.append(drag_line_tag)

    def _update_peaks_on_drag_callback(self, sender: str) -> None:
        alias = dpg.get_item_alias(sender)
        self._edited_drag_lines[alias] = dpg.get_value(sender) * self._SIGNAL_FREQUENCY
        dpg.set_item_label(self._update_peaks_btn_tag, '*Применить')
        dpg.show_item(self._cancel_btn_tag)

    def _update_plots_on_start_change_callback(
        self, sender: str, start_input: float,
    ) -> None:
        self._cancel_changes()

        start = self.sth(start_input)
        length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
        stop = start + length
        channel = dpg.get_value(self._editable_channel_input_tag) - 1

        self._update_editable_signal(start, stop, channel)
        self._update_drag_lines(start, stop, channel)
        self._update_top_signal(start, stop)

    def _make_step_callback(
        self, sender: str, app_data: None, user_data: Direction,
    ) -> None:
        self._cancel_changes()

        match (user_data):
            case Direction.LEFT:
                op = sub
            case _:
                op = add

        cur_pos = self.sth(dpg.get_value(self._start_input_tag))
        step = self.sth(dpg.get_value(self._step_size_tag))
        new_pos = max(0, op(cur_pos, step))
        length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
        stop = new_pos + length
        channel = dpg.get_value(self._editable_channel_input_tag) - 1

        dpg.set_value(
            self._start_input_tag,
            new_pos / self._SIGNAL_FREQUENCY,
        )

        self._update_editable_signal(new_pos, stop, channel)
        self._update_top_signal(new_pos, stop)
        self._update_drag_lines(new_pos, stop, channel)

    def _update_editable_signal_callback(self, sender: str, channel: int) -> None:
        self._cancel_changes()

        start = self.sth(dpg.get_value(self._start_input_tag))
        length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
        stop = start + length

        self._update_editable_signal(start, stop, channel - 1)
        self._update_drag_lines(start, stop, channel - 1)

    def _update_editable_signal(self, start: int, stop: int, channel: int) -> None:
        if self._signal is None:
            return

        signal_cutout = self._signal[channel][start:stop]
        xs = [(start + x) / self._SIGNAL_FREQUENCY for x in range(len(signal_cutout))]

        dpg.set_value(self._editable_channel_series_tag, [xs, signal_cutout])
        dpg.fit_axis_data(self._bottom_signal_y_axis_tag)
        dpg.fit_axis_data(self._bottom_signal_x_axis_tag)

    def _update_top_signal(self, start: int, stop: int) -> None:
        if self._signal is None:
            return

        for channel, series in self._channel_top_series_tags.items():
            signal_cutout = self._signal[channel][start:stop]
            xs = [(start + x) / self._SIGNAL_FREQUENCY for x in range(len(signal_cutout))]

            dpg.set_value(series, [xs, signal_cutout])

        dpg.fit_axis_data(self._top_signal_y_axis_tag)
        dpg.fit_axis_data(self._top_signal_x_axis_tag)

    def _save_as_label_file_callback(self, sender: str, app_data: dict) -> None:
        if self._signal_path is None:
            return

        output_filename = self._signal_path.stem + '.json'

        try:
            folder_path, *__ = app_data['selections'].values()
        except ValueError:
            return

        filepath = Path(folder_path).parent / output_filename
        self._save_label_file(filepath)

    def _save_label_file_callback(self) -> None:
        if not self._inference_result_path:
            return

        self._save_label_file(self._inference_result_path)

    def _save_label_file(self, filepath: Path) -> None:
        if self._signal_path is None:
            return

        if not (path_to_model := self._model_path):
            path_to_model = Path()

        inference_result = InferenceResult(
            peaks=self._peaks,
            meta=InferenceMeta(
                threshold=dpg.get_value(self._threshold_input_tag),
                path_to_model=path_to_model.as_posix(),
                path_to_signal=self._signal_path.as_posix(),
            ),
        )

        with open(filepath, 'w') as out:
            json.dump(inference_result.model_dump(), out)

    def _load_model_callback(self, sender: str, app_data: dict) -> None:
        try:
            filepath, *__ = app_data['selections'].values()
        except ValueError:
            return

        path = Path(filepath)
        self._model_path = path
        dpg.set_value(self._model_label_tag, path.as_posix())

        providers = [
            (
                'CUDAExecutionProvider',
                {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': dpg.get_value(self._gpu_memory_int_field_tag)
                    * 1024
                    * 1024
                    * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                },
            ),
            'CPUExecutionProvider',
        ]

        self._model = OnnxModelWrapper(path, providers)

    def _predict_callback(self, sender: str) -> None:
        if not self._model or not self._signal:
            return

        threshold = dpg.get_value(self._threshold_input_tag)

        compressor = Compressor(
            target_frequency=dpg.get_value(self._target_frequency_tag) * 1000,
        )
        signal_processor = SignalProcessor(
            self._model,
            dpg.get_value(self._batch_size_input_tag),
            compressor,
            threshold=threshold,
        )

        peaks = signal_processor.process(self._signal)

        if dpg.get_value(self._postprocessing_checkbox_tag):
            peaks = postprocess_predictions(
                peaks, dpg.get_value(self._postprocessing_window_width_tag),
            )

        self._peaks = peaks

    def _channels_checkboxes_callback(
        self, sender: str, toggled: bool, channel_no: int,
    ) -> None:
        if self._signal is None:
            return

        top_plot_tag = self._channel_top_series_tags.pop(channel_no, None)

        if not toggled and top_plot_tag:
            dpg.delete_item(top_plot_tag)
            dpg.fit_axis_data(self._top_signal_y_axis_tag)
            dpg.fit_axis_data(self._top_signal_x_axis_tag)

            return

        top_series_tag = generate_uuid()
        start = self.sth(dpg.get_value(self._start_input_tag))
        stop = start + self.sth(dpg.get_value(self._signal_cutout_length_tag))
        signal_cutout = self._signal[channel_no][start:stop]

        dpg.add_line_series(
            [(start + x) / self._SIGNAL_FREQUENCY for x in range(len(signal_cutout))],
            list(signal_cutout),
            tag=top_series_tag,
            label=f'{channel_no + 1} канал',
            parent=self._top_signal_y_axis_tag,
        )

        dpg.fit_axis_data(self._top_signal_y_axis_tag)
        dpg.fit_axis_data(self._top_signal_x_axis_tag)

        self._channel_top_series_tags[channel_no] = top_series_tag

    def _generate_channel_checkboxes(self, num: int) -> None:
        for checkbox in self._channel_checkbox_tags:
            dpg.delete_item(checkbox)

        checkboxes_tags = [generate_uuid() for __ in range(num)]
        self._channel_checkbox_tags = checkboxes_tags

        for channel_no, checkbox in enumerate(self._channel_checkbox_tags, start=1):
            dpg.add_checkbox(
                label=f'{channel_no}',
                callback=self._channels_checkboxes_callback,
                user_data=channel_no - 1,
                tag=checkbox,
                parent=self._left_column_tag,
            )

    def _open_signal_file(self, sender: str, app_data: dict) -> None:
        try:
            filepath, *__ = app_data['selections'].values()
        except ValueError:
            return

        path = Path(filepath)
        self._signal_path = path
        self._signal = np.load(path, mmap_mode='r')
        dpg.configure_item(self._top_signal_plot_tag, label=path.as_posix())

        self._generate_channel_checkboxes(len(self._signal))

        if self._peaks:
            start = self.sth(dpg.get_value(self._start_input_tag))
            length = self.sth(dpg.get_value(self._signal_cutout_length_tag))
            stop = start + length
            channel = dpg.get_value(self._editable_channel_input_tag) - 1

            self._update_editable_signal(start, length, channel)
            self._update_drag_lines(start, stop, channel)


def main() -> None:
    Main(scale=1.0).start_app()


if __name__ == '__main__':
    raise SystemExit(main())
