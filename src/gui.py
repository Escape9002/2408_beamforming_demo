from pathlib import Path
from sys import path as sys_path

project_root_path = Path(Path(__file__).parent, '..')
sys_path.append(str(project_root_path.resolve()))

from pathlib import Path
from PIL import Image
import tkinter
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import pdf2image

from src.config.config import Config
from src.utils.satellite_functions import get_steering_vec, get_aods
from src.utils.get_width_rescale_constant_aspect_ratio import get_width_rescale_constant_aspect_ratio


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_relative_font_size(font_size, window_height_px) -> float:
    """
    pt to px conversion
    """
    return font_size * (96 / 72) / window_height_px

def px_to_pt(pt) -> float:
    return pt * 72 / 96


class BeamformingPlot:

    def __init__(self):

        self.language = 'en'

        self._load_palettes()
        self._load_strings(self.language)

        # get window info todo there has to be a better way to get this data
        root = tkinter.Tk()
        root.withdraw()
        self.window_width, self.window_height = root.winfo_screenwidth(), root.winfo_screenheight()

        self.config = Config()

        self.own_spherical_coordinates = np.array([10, np.pi / 2, np.pi / 2])
        self.users_spherical_coordinates = [
            np.array([1, np.pi / 2, np.pi / 2 + 0.2]),
            # np.array([1, np.pi/2, np.pi/2]),
            np.array([1, np.pi / 2, np.pi / 2 - 0.2]),
        ]
        self.user_aods = get_aods(self.own_spherical_coordinates, self.users_spherical_coordinates)

        self.user_num = len(self.users_spherical_coordinates)
        self.antenna_num = 2

        self.logo_img_height = 100  # in pixels
        self.button_width = 0.1  # relative
        self.button_height = 0.1  # relative
        self.button_pad_horizontal = 30 / self.window_width  # relative
        self.button_pad_vertical = 30 / self.window_height  # relative
        self.logo_img_height_relative = self.logo_img_height / self.window_height

        self.scenario_image_width = 0.20  # relative
        self.scenario_image_height = 0.20  # relative

        self.overlap_plot_height = 0.10  # relative
        self.slider_height = 0.05  # relative, slider uses half available height
        self.slider_pad_vertical = 0.00  # relative


        self.slider_args = {
            'valmin': 0,
            'valmax': 2,
            'valinit': 0,
            'valfmt': '%.2f $\pi$',
            'initcolor': 'none',
            # 'edgecolor': 'black',
            'track_color': '#f2f2f2',
        }
        self.button_args = {
            'color': 'white',
        }

        self.aod_range = np.linspace(self.user_aods[0] - 0.05, self.user_aods[-1] + 0.05, 1000)
        self.user_aod_range_idx = []
        for user_id in range(self.user_num):
            self.user_aod_range_idx.append(np.argmin(np.abs(self.user_aods[user_id] - self.aod_range)))

        self.colors = [self.cp3['blue2'], self.cp3['red2'], 'black']

        self.font_main = {
            'family': 'sans-serif',
            # 'weight': 'bold',
            'size': int(px_to_pt(0.025*self.window_height)),
        }
        self.relative_main_font_size = get_relative_font_size(self.font_main['size'], self.window_height)

        self.font_title = {
            'size': int(px_to_pt(0.05*self.window_height)),
            'weight': 'bold',
            # 'color': self.cp3['blue2'],
        }
        self.relative_title_font_size = get_relative_font_size(self.font_title['size'], self.window_height)

        mpl.rc('font', **self.font_main)
        mpl.rc('lines', linewidth=2)
        mpl.rcParams['toolbar'] = 'None'

        self.fig, self.axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        self.fig.canvas.manager.full_screen_toggle()

        self.axes[0].set_ylim([0, 2.2])
        self.axes[1].set_ylim([0, 2.5])

        for ax in self.axes:
            ax.set_xlim([self.aod_range[0], self.aod_range[-1]])

        for ax in self.axes:
            ax.grid(visible=True, axis='y')

        self.axes[1].set_xticks([])

        self.ax_overlapplot = self.axes[0].inset_axes((0.06, 0.6, 0.2, 0.3))
        self.lines_overlapplot = []

        self.ax_overlapplot.grid(visible=True, axis='y')
        self.ax_overlapplot.set_xticks([])
        self.ax_overlapplot.set_yticks([])

        self.result_text = self.axes[1].text(
            self.user_aods[-1]*1.015, 2.0, '',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round',
            )
        )

        self.axes[0].indicate_inset(
            bounds=(self.user_aods[0] - 0.001, 0, 0.002, 0.1),
            inset_ax=self.ax_overlapplot,
            edgecolor='black',
        )

        # set title
        self.title_text = self.fig.text(
            self.button_pad_horizontal,
            1 - self.button_pad_vertical - self.logo_img_height_relative - self.relative_title_font_size - self.button_pad_vertical/2,
            '',
            fontdict=self.font_title,
        )
        self.subtitle_text = self.fig.text(
            self.button_pad_horizontal,
            1 - self.button_pad_vertical - self.logo_img_height_relative - self.relative_title_font_size - self.button_pad_vertical/2 - self.relative_main_font_size - self.button_pad_vertical/4,
            '',
            fontdict=self.font_main,
        )

        self.scenario_images = {
            '2-2': pdf2image.convert_from_path(Path(Path.cwd(), 'src', '2-2.pdf'))[0],
            '2-3': pdf2image.convert_from_path(Path(Path.cwd(), 'src', '2-3.pdf'))[0],
            '2-4': pdf2image.convert_from_path(Path(Path.cwd(), 'src', '2-4.pdf'))[0],
            '3-2': pdf2image.convert_from_path(Path(Path.cwd(), 'src', '3-2.pdf'))[0],
            '3-3': pdf2image.convert_from_path(Path(Path.cwd(), 'src', '3-3.pdf'))[0],
            '3-4': pdf2image.convert_from_path(Path(Path.cwd(), 'src', '3-4.pdf'))[0],
        }
        self.scenario_figure_axis = self.fig.add_axes(
            (
                1 - self.scenario_image_width - 3 * self.button_width - self.button_pad_horizontal,
                1 - self.scenario_image_height - self.button_pad_vertical,
                self.scenario_image_width,
                self.scenario_image_height,
            ),
        )
        self.scenario_figure_axis.axis('off')
        self.scenario_figure = None

        # place logos
        logo_paths = [
            Path(Path.cwd(), 'src', 'logo_unilogo.png'),
            Path(Path.cwd(), 'src', 'logo_ANT.png'),
        ]
        logo_images = [
            Image.open(logo_path)
            for logo_path in logo_paths
        ]
        for logo_id, logo_image in enumerate(logo_images):
            self.fig.figimage(
                logo_image.resize((
                    get_width_rescale_constant_aspect_ratio(logo_image, self.logo_img_height),
                    self.logo_img_height,
                )), xo=15 + logo_id * 300, yo=int(self.window_height - self.logo_img_height - 15)
            )

        self.lines_power_gain = []
        self.lines_signal_to_interference = []

        self.slider_axes = []
        self.sliders = []

        self.text_user_pos = []
        self.text_user_antennas = []

        self.ax_button_2ant = self.fig.add_axes((1 - 3 * self.button_width - self.button_pad_horizontal,
                                                 1 - self.button_height - self.button_pad_vertical, self.button_width,
                                                 self.button_height))
        self.ax_button_3ant = self.fig.add_axes((1 - 2 * self.button_width - self.button_pad_horizontal,
                                                 1 - self.button_height - self.button_pad_vertical, self.button_width,
                                                 self.button_height))
        self.ax_button_4ant = self.fig.add_axes((1 - 1 * self.button_width - self.button_pad_horizontal,
                                                 1 - self.button_height - self.button_pad_vertical, self.button_width,
                                                 self.button_height))
        self.ax_button_ai_solution = self.fig.add_axes((1 - 1 * self.button_width - self.button_pad_horizontal,
                                                        1 - 2 * self.button_height - self.button_pad_vertical,
                                                        self.button_width, self.button_height))
        self.ax_button_language_toggle = self.fig.add_axes((1 - 2 * self.button_width - self.button_pad_horizontal,
                                                            1 - 2 * self.button_height - self.button_pad_vertical,
                                                            self.button_width, self.button_height))
        self.ax_button_user_toggle = self.fig.add_axes((1 - 3 * self.button_width - self.button_pad_horizontal,
                                                        1 - 2 * self.button_height - self.button_pad_vertical,
                                                        self.button_width, self.button_height))
        self.button_2_ant = Button(self.ax_button_2ant, '', **self.button_args)
        self.button_3_ant = Button(self.ax_button_3ant, '', **self.button_args)
        self.button_4_ant = Button(self.ax_button_4ant, '', **self.button_args)
        self.button_ai_solution = Button(self.ax_button_ai_solution, '', **self.button_args)
        self.button_language_toggle = Button(self.ax_button_language_toggle, '', **self.button_args)
        self.button_user_toggle = Button(self.ax_button_user_toggle, '', **self.button_args)
        self.button_2_ant.on_clicked(self.build_2_ant)
        self.button_3_ant.on_clicked(self.build_3_ant)
        self.button_4_ant.on_clicked(self.build_4_ant)
        self.button_ai_solution.on_clicked(self.solve)
        self.button_language_toggle.on_clicked(self.toggle_language)
        self.button_user_toggle.on_clicked(self.toggle_user_number)

        self.fig.subplots_adjust(top=0.74, right=0.65, left=0.1, bottom=0.08)

        self.build_plot()

    def clear_plot(
            self,
    ) -> None:
        """
        Delete non-static elements of the figure
        """

        for slider_axes_user in self.slider_axes:
            for ax in slider_axes_user:
                ax.remove()
                del ax
        del self.slider_axes
        self.slider_axes = []

        for sliders_user in enumerate(self.sliders):
            for slider in enumerate(sliders_user):
                del slider
        del self.sliders
        self.sliders = []

        for ax_id in range(2):
            while len(self.axes[ax_id].get_lines()) > 0:
                line = self.axes[ax_id].get_lines().pop(0).remove()
                del line

        for ax_id in range(2):
            while len(self.axes[ax_id].collections) > 0:
                for collection in self.axes[ax_id].collections:
                    collection.remove()
                    del collection

        while len(self.ax_overlapplot.get_lines()) > 0:
            line = self.ax_overlapplot.get_lines().pop(0).remove()
            del line
        self.lines_overlapplot = []

        if self.scenario_figure:
            self.scenario_figure.remove()
            del self.scenario_figure
            self.scenario_figure = None

        for text in self.text_user_pos:
            text.remove()
            del text
        del self.text_user_pos
        self.text_user_pos = []

        for text in self.text_user_antennas:
            text.remove()
            del text
        del self.text_user_antennas
        self.text_user_antennas = []

    def build_plot(
            self,
    ) -> None:
        """
        Arrange non-static elements of the plot
        """

        self.clear_plot()

        # set scenario figure
        self.scenario_figure = self.scenario_figure_axis.imshow(self.scenario_images[f'{self.user_num}-{self.antenna_num}'])

        # set main axes lines
        w_precoder = np.exp(1j * np.zeros((self.antenna_num, self.user_num)))

        power_gains_users, signal_to_interference_ratio_per_user = self.calculate_data(w_precoder)
        self.lines_power_gain = []
        self.lines_signal_to_interference = []
        for user_id in range(self.user_num):
            self.lines_power_gain.append(
                self.axes[0].plot(self.aod_range, power_gains_users[user_id, :], color=self.colors[user_id])[0])
            self.lines_signal_to_interference.append(
                self.axes[1].plot(self.aod_range, signal_to_interference_ratio_per_user[user_id, :],
                                  color=self.colors[user_id])[0])

        # set result text
        sum_sinr = sum(np.maximum(0, np.diagonal(signal_to_interference_ratio_per_user[:, self.user_aod_range_idx])))
        self.result_text.set_text(f'{sum_sinr:.2f}')

        # mark user positions
        for user_id, user_aod in enumerate(self.user_aods):
            for ax in self.axes:
                s = ax.scatter(user_aod, 0, color=self.colors[user_id], s=60)
                s.set_clip_on(False)
            # s = self.fig.text(0.5, 0.5, s=f'User {user_id}', color=self.colors[user_id])
            self.text_user_pos.append(
                self.axes[0].text(user_aod, -0.15, s='', color=self.colors[user_id],
                                  verticalalignment='top', horizontalalignment='center'))

            self.axes[0].vlines(user_aod, 0, 20, ls=':', color=self.colors[user_id])
            self.axes[1].vlines(user_aod, 0, 15, ls=':', color=self.colors[user_id])

        # create lines for wave overlap inset plot
        angles = self.calculate_gain_at_userpos(user_id=0, w_precoder=np.ones(self.antenna_num)[np.newaxis])
        for angle in angles:
            self.lines_overlapplot.append(
                self.ax_overlapplot.plot(np.linspace(0, 2 * np.pi, 100), np.sin(np.linspace(0, 2 * np.pi, 100) - angle),
                                         color=self.colors[0])[0])

        # place user texts for antenna sliders
        for user_id in range(self.user_num):
            self.text_user_antennas.append(
                self.fig.text(
                    x=1 - 3 * self.button_width - self.button_pad_horizontal,
                    y=(
                            1 - self.button_pad_vertical - 2 * self.button_height - self.button_pad_vertical
                            - (user_id+1) * self.relative_main_font_size
                            - user_id * self.antenna_num * self.slider_height
                            - user_id * .5 * self.relative_main_font_size  # extra inter-user pad
                    ),
                    s='',
                    color=self.colors[user_id],
                    va='bottom',
                )
            )

        # create sliders
        self.slider_axes = [
            [
                self.fig.add_axes((
                    1 - 3 * self.button_width - self.button_pad_horizontal + 0.1,
                    (
                            1 - self.button_pad_vertical - 2 * self.button_height - self.button_pad_vertical
                            - (user_id+1) * self.relative_main_font_size
                            - (antenna_id+1) * self.slider_height

                            - user_id * self.antenna_num * self.slider_height
                            - user_id * .5 * self.relative_main_font_size  # extra inter-user pad
                    ),
                    0.15,
                    self.slider_height,
                ))
                for antenna_id in range(self.antenna_num)
            ]
            for user_id in range(self.user_num)
        ]

        self.sliders = [
            [
                Slider(
                    ax=self.slider_axes[user_id][antenna_id],
                    label='',
                    facecolor=self.colors[user_id],
                    handle_style={'facecolor': self.colors[user_id]},
                    **self.slider_args
                )
                for antenna_id in range(self.antenna_num)
            ]
            for user_id in range(self.user_num)
        ]

        for sliders_user in self.sliders:
            for slider in sliders_user:
                slider.on_changed(self.update_plots)

        self.set_strings()

        self.fig.canvas.draw_idle()

    def calculate_gain_at_userpos(
            self,
            user_id: int,
            w_precoder: np.ndarray,
    ) -> np.ndarray:

        user_aod = self.user_aods[user_id]
        steering_vec = get_steering_vec(
            antenna_num=self.antenna_num,
            antenna_distance=self.config.sat_ant_dist,
            wavelength=self.config.wavelength,
            cos_aod=np.cos(user_aod),
        )

        norm_factor = np.sqrt(1 / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
        normalized_precoder = norm_factor * w_precoder

        return np.angle(steering_vec * normalized_precoder[:, user_id])

    def calculate_data(
            self,
            w_precoder: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        norm_factor = np.sqrt(1 / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
        normalized_precoder = norm_factor * w_precoder

        power_gains_users = np.zeros((self.user_num, len(self.aod_range)))
        signal_to_interference_ratio_per_user = np.zeros((self.user_num, len(self.aod_range)))
        for aod_id, aod in enumerate(self.aod_range):
            steering_vec = get_steering_vec(antenna_num=self.antenna_num, antenna_distance=self.config.sat_ant_dist,
                                            wavelength=self.config.wavelength, cos_aod=np.cos(aod))
            for user_id in range(len(self.users_spherical_coordinates)):
                power_gain_user = abs(np.matmul(steering_vec, normalized_precoder[:, user_id])) ** 2
                power_gains_users[user_id, aod_id] = power_gain_user

        for user_id in range(self.user_num):
            signal_to_interference_ratio_per_user[user_id, :] = (
                power_gains_users[user_id, :] / (
                    np.sum(np.delete(power_gains_users, user_id, axis=0), axis=0)
                    + 0.01  # regularizing noise
                )
            )

        return power_gains_users, np.log10(signal_to_interference_ratio_per_user)

    def update_plots(self, val):
        w_precoder = np.exp(1j * np.zeros((self.antenna_num, self.user_num)))
        for sliders_user_id, sliders_user in enumerate(self.sliders):
            for slider_id, slider in enumerate(sliders_user):
                w_precoder[slider_id, sliders_user_id] = np.exp(1j * slider.val * np.pi)

        power_gains_users, signal_to_interference_ratio_per_user = self.calculate_data(w_precoder)
        for line_id, line in enumerate(self.lines_power_gain):
            line.set_ydata(power_gains_users[line_id, :])
        for line_id, line in enumerate(self.lines_signal_to_interference):
            line.set_ydata(signal_to_interference_ratio_per_user[line_id, :])

        sum_sinr = sum(
            np.maximum(
                0,
                np.diagonal(signal_to_interference_ratio_per_user[:, self.user_aod_range_idx])
            )
        )
        self.result_text.set_text(f'{sum_sinr:.2f}')

        angles = self.calculate_gain_at_userpos(user_id=0, w_precoder=w_precoder)
        for angle, line in zip(angles, self.lines_overlapplot):
            line.set_ydata(np.sin(np.linspace(0, 2 * np.pi, 100) - angle))

        self.fig.canvas.draw_idle()

    def build_2_ant(
            self,
            event,
    ) -> None:
        self.antenna_num = 2
        self.build_plot()

    def build_3_ant(
            self,
            event,
    ) -> None:
        self.antenna_num = 3
        self.build_plot()

    def build_4_ant(
            self,
            event,
    ) -> None:
        self.antenna_num = 4
        self.build_plot()

    def solve(
            self,
            event,
    ) -> None:
        """
        Pre-computed hard coded solutions
        """

        if self.user_num == 2:
            if self.antenna_num == 2:
                vals = np.array([1.33392694, 0.00217974, 2., 1.336326])  # 3.7434581600273287

            elif self.antenna_num == 3:
                vals = np.array([1.43766934, 0.32953251, 1.77591245, 1.9991804, 0.11070739, 1.65571502])  # 4.348553352347908

            elif self.antenna_num == 4:
                vals = np.array([1.65264511e+00, 3.45269947e-01, 1.99999797e+00, 1.82675364e-06, 1.99985408e+00, 1.98439029e+00, 3.54067908e-01, 1.65027350e+00])  # 4.364126149316508

        elif self.user_num == 3:
            if self.antenna_num == 2:
                vals = np.array([0.17, 1.11987319, 0.06849515, 0.49781637, 0.45381053, 1.40067422])  # 1.82, all power left
                # vals = np.array([0.27038586, 0.10755455, 0.310714, 0.9434595, 0.7684472, 0.0134297 ])  # 1.82, all power right

            if self.antenna_num == 3:
                vals = np.array([0.767586  , 1.23849644, 0.0155399 , 1.11047316, 0.2418609, 1.6794695 , 1.44843245, 1.2447759 , 1.33063957])  #  3.98

            if self.antenna_num == 4:
                vals = np.array([1.8236148, 0.6236967, 0.99348598, 0.4835393, 1.30029637, 0.3300958, 0.7820382, 0.3140922, 0.0082575, 1.44706702, 0.9572313 ,1.3421065 ])  # 5.08
                # vals = np.array([0.4544177 , 1.0933318 , 1.1671627 , 1.09039766, 0.4608915 , .4944167 , 1.4022036 , 1.43636626, 0.1882944 , 0.0646899 , 0.77249672, 1.52867377])  # 5.08

        else:
            vals = np.zeros(self.config.sat_nr * self.antenna_num * self.user_num)

        vals = vals.reshape((self.config.sat_nr * self.antenna_num, self.user_num))

        for sliders_user_id, sliders_user in enumerate(self.sliders):
            for slider_id, slider in enumerate(sliders_user):
                slider.set_val(vals[slider_id, sliders_user_id])

    def toggle_language(
            self,
            event,
    ) -> None:

        if self.language == 'en':
            self.language = 'de'
        elif self.language == 'de':
            self.language = 'en'

        self._load_strings(language=self.language)

        self.set_strings()

        self.fig.canvas.draw_idle()

    def toggle_user_number(
            self,
            event,
    ) -> None:

        if len(self.users_spherical_coordinates) == 2:

            self.users_spherical_coordinates = [
                np.array([1, np.pi / 2, np.pi / 2 + 0.2]),
                np.array([1, np.pi / 2, np.pi / 2]),
                np.array([1, np.pi / 2, np.pi / 2 - 0.2]),
            ]

        elif len(self.users_spherical_coordinates) == 3:

            self.users_spherical_coordinates = [
                np.array([1, np.pi / 2, np.pi / 2 + 0.2]),
                # np.array([1, np.pi / 2, np.pi / 2]),
                np.array([1, np.pi / 2, np.pi / 2 - 0.2]),
            ]

        self.user_aods = get_aods(self.own_spherical_coordinates, self.users_spherical_coordinates)

        self.user_num = len(self.users_spherical_coordinates)

        self.user_aod_range_idx = []
        for user_id in range(self.user_num):
            self.user_aod_range_idx.append(np.argmin(np.abs(self.user_aods[user_id] - self.aod_range)))

        self.build_plot()

    def _load_palettes(
            self,
    ) -> None:

        self.cp3: dict[str: str] = {  # uni branding
            'red1': '#9d2246',
            'red2': '#d50c2f',
            'red3': '#f39ca9',
            'blue1': '#00326d',
            'blue2': '#0068b4',
            'blue3': '#89b4e1',
            'purple1': '#3b296a',
            'purple2': '#8681b1',
            'purple3': '#c7c1e1',
            'peach1': '#d45b65',
            'peach2': '#f4a198',
            'peach3': '#fbdad2',
            'orange1': '#f7a600',
            'orange2': '#fece43',
            'orange3': '#ffe7b6',
            'green1': '#008878',
            'green2': '#8acbb7',
            'green3': '#d6ebe1',
            'yellow1': '#dedc00',
            'yellow2': '#f6e945',
            'yellow3': '#fff8bd',
            'white': '#ffffff',
            'black': '#000000',
        }

    def _load_strings(
            self,
            language: str,
    ) -> None:

        with open(Path(Path.cwd(), 'src', f'strings_{language}.json')) as file:
            self.strings = json.load(file)

    def set_strings(
            self,
    ) -> None:

        self.title_text.set_text(self.strings["plot_title"])
        self.subtitle_text.set_text(f'{self.antenna_num} {self.strings["antennapl"]}, {self.user_num} {self.strings["userpl"]}')

        self.axes[1].set_xlabel(self.strings['direction'])
        self.axes[0].set_ylabel(self.strings['power_gain'])
        self.axes[1].set_ylabel(self.strings['signal_strength'])

        self.button_2_ant.label.set_text(f'2 {self.strings["antennapl"]}')
        self.button_3_ant.label.set_text(f'3 {self.strings["antennapl"]}')
        self.button_4_ant.label.set_text(f'4 {self.strings["antennapl"]}')

        self.button_ai_solution.label.set_text(self.strings['ai'])
        self.button_language_toggle.label.set_text(self.strings['language_other'])
        self.button_user_toggle.label.set_text(self.strings["less_users"] if self.user_num == 3 else self.strings["more_users"])

        for user_id, text in enumerate(self.text_user_pos):
            text.set_text(f'{self.strings["user"]} {user_id+1}')

        for user_id, text in enumerate(self.text_user_antennas):
            text.set_text(f'{self.strings["user"]} {user_id+1}')

        for user_id in range(self.user_num):
            for antenna_id in range(self.antenna_num):
                self.sliders[user_id][antenna_id].label.set_text(f'{self.strings["antenna"]} {antenna_id+1}')


if __name__ == '__main__':
    bf = BeamformingPlot()
    plt.show()
