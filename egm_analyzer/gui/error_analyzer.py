import json
import random
from pathlib import Path
from types import SimpleNamespace

import dearpygui.dearpygui as dpg
import numpy as np

from egm_analyzer.types import Error
from egm_analyzer.types import InferenceResult
from egm_analyzer.types import MetricsResult
from egm_analyzer.types import Peak


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if not args and not kwargs:
                raise ValueError('Trying to access not initialized class')

            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SignalView(metaclass=MetaSingleton):
    def __init__(self, metrics_result_path: Path) -> None:
        (
            self._signal,
            self._ground_truth,
            self._predictions,
            self._errors,
        ) = self._load_data(metrics_result_path)

        self._current_error = self._errors[0]
        self.update_by_error()

    def update_data(self, metrics_result_path: Path) -> None:
        (
            self._signal,
            self._ground_truth,
            self._predictions,
            self._errors,
        ) = self._load_data(metrics_result_path)

        self._current_error = self._errors[0]
        self.update_by_error()

    def _load_data(
        self, metrics_result_path: Path,
    ) -> tuple[np.ndarray, list[list[int]], list[list[Peak]], list[Error]]:
        metrics_result = MetricsResult(**json.loads(metrics_result_path.read_text()))
        inference_result_path = Path(metrics_result.meta.inference_result_path)
        inference_result = InferenceResult(
            **json.loads(inference_result_path.read_text())
        )

        signal = np.load(inference_result.meta.path_to_signal, mmap_mode='r')
        ground_truth = json.loads(
            Path(metrics_result.meta.ground_truth_path).read_text(),
        )
        predictions = inference_result.peaks
        errors = metrics_result.errors

        dpg.configure_item(Tags.plot, label=metrics_result_path.stem.split('_')[0])

        return (
            signal,
            ground_truth,
            predictions,
            errors,
        )

    def pick_random_error(self) -> Error:
        return random.choice(self._errors)

    def _plot(self, start: int, length: int, channel: int) -> None:
        dpg.delete_item(Tags.plot, children_only=True, slot=0)
        stop = start + length

        signal_ys = list(self._signal[channel][start:stop])
        signal_xs = list(range(start, stop))

        dpg.set_value(Tags.signal_series, [signal_xs, signal_ys])

        ground_truth = []
        # TODO: Add binary search for speed
        for gt_pos in self._ground_truth[channel]:
            if gt_pos > start:
                if gt_pos > stop:
                    break

                ground_truth.append(float(gt_pos))

        dpg.set_value(Tags.ground_truth_series, [ground_truth])

        search_segments: list[float] = []
        predictions: list[float] = []
        # TODO: Add binary search for speed
        for prediction in self._predictions[channel]:
            if prediction.position > start:
                if prediction.position > stop:
                    break

                predictions.append(float(prediction.position))
                dpg.add_plot_annotation(
                    label=(
                        f'Stage: {prediction.creation_stage} \n'
                        f'Time: {(prediction.position / 5000):.5f}s'
                    ),
                    default_value=(prediction.position, 0),
                    offset=(-15, 15),
                    color=[255, 255, 0, 150],
                    parent=Tags.plot,
                    clamped=False,
                    show=dpg.get_value(Tags.toggle_annotations),
                )
                if prediction.search_segment:
                    search_segments.extend(prediction.search_segment)

        dpg.set_value(Tags.peaks_series, [predictions])
        dpg.set_value(
            Tags.search_segments_series,
            [search_segments, [0 for __ in range(len(search_segments))]],
        )

        errors: list[float] = []
        for error in self._errors:
            if start < error.position < stop and error.channel == channel:
                errors.append(float(error.position))
                dpg.add_plot_annotation(
                    label=f'Error: {error.error_type} \nTime {error.position / 5000:.5f}s',
                    default_value=(error.position, 0),
                    offset=(15, -15),
                    color=[255, 0, 0, 150],
                    parent=Tags.plot,
                    clamped=False,
                    show=dpg.get_value(Tags.toggle_annotations),
                )

        dpg.fit_axis_data(Tags.signal_y_axis)
        dpg.fit_axis_data(Tags.signal_x_axis)

    def update_by_error(self) -> None:
        self._current_error = self.pick_random_error()

        length = dpg.get_value(Tags.length_input)

        start = int(max(0, self._current_error.position - length // 2))

        if start + length >= (
            max_len := len(self._signal[self._current_error.channel])
        ):
            start = max_len - length

        dpg.set_value(Tags.start_input, start)
        dpg.set_value(Tags.channel_input, self._current_error.channel)

        self._plot(start, length, self._current_error.channel)

    def update_by_input(
        self,
        start: int | None = None,
        length: int | None = None,
        channel: int | None = None,
    ) -> None:
        if channel is not None:
            start = dpg.get_value(Tags.start_input)
            length = dpg.get_value(Tags.length_input)
            self._plot(start, length, channel)
        elif start is not None:
            length = dpg.get_value(Tags.length_input)
            channel = dpg.get_value(Tags.channel_input)
            self._plot(start, length, channel)
        elif length is not None:
            channel = dpg.get_value(Tags.channel_input)
            start = int(max(0, self._current_error.position - length // 2))

            if start + length >= (
                max_len := len(self._signal[self._current_error.channel])
            ):
                start = max_len - length

            self._plot(start, length, channel)


class Tags(SimpleNamespace):
    main_window = 'main_window_id'

    plot = 'plot_id'

    signal_series = 'signal_series_id'
    peaks_series = 'peaks_series_id'
    ground_truth_series = 'ground_truth_id'
    search_segments_series = 'search_segments_series_id'

    signal_y_axis = 'signal_y_axis_id'
    signal_x_axis = 'signal_x_axis_id'

    # Themes
    ground_truth_theme = 'ground_truth_theme'
    prediction_theme = 'prediction_theme'

    # File menu
    pick_inference_result_file_dialog = 'pick_inference_result_file_dialog_id'

    # Settings menu
    toggle_annotations = 'toggle_annotations_id'

    # Input fields
    length_input = 'length_input_id'
    channel_input = 'channel_input_id'
    start_input = 'start_input_id'


def metrics_result_file_pick_callback(sender: str, app_data: dict) -> None:
    try:
        filepath, *__ = app_data['selections'].values()
    except ValueError:
        return

    path = Path(filepath)

    try:
        SignalView().update_data(path)
    except ValueError:
        SignalView(Path(filepath))


def next_error_callback(sender: str, app_data: dict) -> None:
    SignalView().update_by_error()  # type: ignore


def channel_update_callback(sender: str, app_data: dict) -> None:
    channel = dpg.get_value(Tags.channel_input)
    SignalView().update_by_input(channel=channel)


def start_update_callback(sender: str, app_data: dict) -> None:
    start = dpg.get_value(Tags.start_input)
    SignalView().update_by_input(start=start)


def length_update_callback(sender: str, app_data: dict) -> None:
    length = dpg.get_value(Tags.length_input)
    SignalView().update_by_input(length=length)


def toggle_annotations(sender: str, show_annotations: bool) -> None:
    annotations: list[int] = dpg.get_item_children(Tags.plot, slot=0)

    if show_annotations:
        for annotation in annotations:
            dpg.show_item(annotation)

        return

    for annotation in annotations:
        dpg.hide_item(annotation)


def main() -> int:
    tags = Tags()

    dpg.create_context()

    with dpg.file_dialog(
        show=False,
        callback=metrics_result_file_pick_callback,
        tag=tags.pick_inference_result_file_dialog,
        width=700,
        height=400,
    ):
        dpg.add_file_extension('.json', color=(0, 255, 0, 255), custom_text='[Json]')

    with dpg.window(tag=tags.main_window):
        with dpg.theme(tag=tags.ground_truth_theme):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots,
                )

            with dpg.theme_component(dpg.mvVLineSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (0, 255, 0, 100), category=dpg.mvThemeCat_Plots,
                )

        with dpg.theme(tag=tags.prediction_theme):
            with dpg.theme_component(dpg.mvVLineSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (255, 167, 0, 100),
                    category=dpg.mvThemeCat_Plots,
                )

        with dpg.menu_bar():
            with dpg.menu(label='File'):
                dpg.add_menu_item(
                    label='Open',
                    callback=lambda: dpg.show_item(
                        tags.pick_inference_result_file_dialog,
                    ),
                )

            with dpg.menu(label='Settings'):
                dpg.add_menu_item(
                    label='Show annotations',
                    check=True,
                    default_value=True,
                    callback=toggle_annotations,
                    tag=tags.toggle_annotations,
                )

        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_input_int(
                    label='Channel',
                    min_value=0,
                    max_value=63,
                    min_clamped=True,
                    max_clamped=True,
                    width=150,
                    step_fast=8,
                    tag=tags.channel_input,
                    callback=channel_update_callback,
                )
                dpg.add_input_int(
                    label='Length',
                    default_value=10_000,
                    width=150,
                    min_value=0,
                    min_clamped=True,
                    tag=tags.length_input,
                    callback=length_update_callback,
                )
                dpg.add_input_int(
                    label='Start',
                    width=150,
                    min_value=0,
                    min_clamped=True,
                    step_fast=dpg.get_value(tags.length_input) // 5,
                    tag=tags.start_input,
                    callback=start_update_callback,
                )

            dpg.add_button(
                label='Next Error',
                callback=next_error_callback,
            )

        with dpg.plot(width=-1, height=-1, anti_aliased=True, tag=tags.plot):
            # optionally create legend
            dpg.add_plot_legend()

            # REQUIRED: create x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label='x', tag=tags.signal_x_axis)
            dpg.add_plot_axis(dpg.mvYAxis, label='y', tag=tags.signal_y_axis)

            # series belong to a y axis
            dpg.add_line_series(
                [],
                [],
                label='Signal',
                parent=tags.signal_y_axis,
                tag=tags.signal_series,
            )
            dpg.add_vline_series(
                [],
                parent=tags.signal_y_axis,
                label='Predicted Peak',
                tag=tags.peaks_series,
            )
            dpg.add_vline_series(
                [],
                parent=tags.signal_y_axis,
                label='Ground Truth',
                tag=tags.ground_truth_series,
            )
            dpg.add_scatter_series(
                [],
                [],
                parent=tags.signal_y_axis,
                label='Search Segment',
                tag=tags.search_segments_series,
            )

            dpg.bind_item_theme(tags.signal_series, tags.ground_truth_theme)
            dpg.bind_item_theme(tags.peaks_series, tags.prediction_theme)
            dpg.bind_item_theme(tags.ground_truth_series, tags.ground_truth_theme)

    dpg.create_viewport(title='Error Analyzer', width=1000, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(tags.main_window, True)
    dpg.start_dearpygui()
    dpg.destroy_context()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
