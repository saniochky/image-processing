import image_convolution
import haar_wavelet
import rank_approx
import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.image import Image


class ImageApp(App):
    def __init__(self):
        super().__init__()
        self.outer_layout = BoxLayout(padding=10, orientation='vertical')
        self.first_in_layout = BoxLayout(padding=10, orientation='horizontal', size_hint=(1, 0.5))
        self.second_in_layout = BoxLayout(padding=10, orientation='horizontal', size_hint=(1, 0.5))
        self.third_in_layout = BoxLayout(padding=10, orientation='horizontal', size_hint=(1, 2))
        self.svd_rank_layout = AnchorLayout(anchor_x="right", anchor_y="top")
        self.svd_number_layout = AnchorLayout(anchor_x="left", anchor_y="top")
        self.hw_ratio_layout = AnchorLayout(anchor_x="right", anchor_y="top")
        self.hw_number_layout = AnchorLayout(anchor_x="left", anchor_y="top")
        self.hw_normalization_layout = AnchorLayout(anchor_x="center", anchor_y="top")
        self.rgb_layout_1 = AnchorLayout(anchor_x="center", anchor_y="top")
        self.rgb_layout_2 = AnchorLayout(anchor_x="center", anchor_y="top")
        self.conv_choice_layout = BoxLayout(padding=0, orientation='vertical')
        self.conv_dd_layout = AnchorLayout(anchor_x="center", anchor_y="top")
        self.image_layout = AnchorLayout(anchor_x="center", anchor_y="center")
        self.reset_layout = AnchorLayout(anchor_x="right", anchor_y="center", size_hint=(0.1, 1))

        self.normalization_dropdown = DropDown()
        self.rgb_dropdown_1 = DropDown()
        self.rgb_dropdown_2 = DropDown()
        self.blur_dropdown = DropDown()
        self.e_det_dropdown = DropDown()

        self.svd_button = Button(text="SVD", on_press=self.on_svd)
        self.haar_wavelet_button = Button(text="Haar Wavelet", on_press=self.on_hw)
        self.convolution_button = Button(text="Image Convolution", on_press=self.on_conv)
        self.reset_button = Button(text="Reset", on_press=self.reset, size_hint_y=None, height=40)
        self.normalization_button = Button(text="Normalization", size_hint=(0.6, 0.6))
        self.rgb_button_1 = Button(text="RGB", size_hint=(0.6, 0.6))
        self.rgb_button_2 = Button(text="RGB", size_hint=(0.6, 0.6))
        self.blur_button = Button(text="Blur", on_press=self.on_blur)
        self.sharpen_button = Button(text="Sharpen", on_press=self.on_sharpen)
        self.e_det_button = Button(text="Edge Detection", on_press=self.on_e_det)
        self.type_button_1 = Button(text="Type", size_hint=(0.6, 0.6))
        self.type_button_2 = Button(text="Type", size_hint=(0.6, 0.6))

        self.yes_button_1 = Button(text="Yes", size_hint_y=None, height=25, on_press=self.normalization_true)
        self.yes_button_1.bind(on_release=lambda btn: self.normalization_dropdown.select(btn.text))
        self.normalization_dropdown.add_widget(self.yes_button_1)
        self.no_button_1 = Button(text="No", size_hint_y=None, height=25, on_press=self.normalization_false)
        self.no_button_1.bind(on_release=lambda btn: self.normalization_dropdown.select(btn.text))
        self.normalization_dropdown.add_widget(self.no_button_1)

        self.yes_button_2 = Button(text="Yes", size_hint_y=None, height=25, on_press=self.svd_rgb_true)
        self.yes_button_2.bind(on_release=lambda btn: self.rgb_dropdown_1.select(btn.text))
        self.rgb_dropdown_1.add_widget(self.yes_button_2)
        self.no_button_2 = Button(text="No", size_hint_y=None, height=25, on_press=self.svd_rgb_false)
        self.no_button_2.bind(on_release=lambda btn: self.rgb_dropdown_1.select(btn.text))
        self.rgb_dropdown_1.add_widget(self.no_button_2)

        self.yes_button_3 = Button(text="Yes", size_hint_y=None, height=25, on_press=self.hw_rgb_true)
        self.yes_button_3.bind(on_release=lambda btn: self.rgb_dropdown_2.select(btn.text))
        self.rgb_dropdown_2.add_widget(self.yes_button_3)
        self.no_button_3 = Button(text="No", size_hint_y=None, height=25, on_press=self.hw_rgb_false)
        self.no_button_3.bind(on_release=lambda btn: self.rgb_dropdown_2.select(btn.text))
        self.rgb_dropdown_2.add_widget(self.no_button_3)

        self.regular_button = Button(text="Regular Blur", size_hint_y=None, height=25, on_press=self.blur_true)
        self.regular_button.bind(on_release=lambda btn: self.blur_dropdown.select(btn.text))
        self.blur_dropdown.add_widget(self.regular_button)
        self.gaussian_button = Button(text="Gaussian Blur", size_hint_y=None, height=25, on_press=self.blur_false)
        self.gaussian_button.bind(on_release=lambda btn: self.blur_dropdown.select(btn.text))
        self.blur_dropdown.add_widget(self.gaussian_button)

        self.normalization_button.bind(on_release=self.normalization_dropdown.open)
        self.normalization_dropdown.bind(on_select=lambda ins, x: setattr(self.normalization_button, 'text', x))
        self.rgb_button_1.bind(on_release=self.rgb_dropdown_1.open)
        self.rgb_dropdown_1.bind(on_select=lambda ins, x: setattr(self.rgb_button_1, 'text', x))
        self.rgb_button_2.bind(on_release=self.rgb_dropdown_2.open)
        self.rgb_dropdown_2.bind(on_select=lambda ins, x: setattr(self.rgb_button_2, 'text', x))

        self.vertical_e_button = Button(text="Vertical Edges", size_hint_y=None, height=25, on_press=self.e_det_true)
        self.vertical_e_button.bind(on_release=lambda btn: self.e_det_dropdown.select(btn.text))
        self.e_det_dropdown.add_widget(self.vertical_e_button)
        self.horizontal_e_button = Button(text="Horizontal Edges", size_hint_y=None, height=25, on_press=self.e_det_false)
        self.horizontal_e_button.bind(on_release=lambda btn: self.e_det_dropdown.select(btn.text))
        self.e_det_dropdown.add_widget(self.horizontal_e_button)

        self.type_button_1.bind(on_release=self.blur_dropdown.open)
        self.blur_dropdown.bind(on_select=lambda ins, x: setattr(self.type_button_1, 'text', x))
        self.type_button_2.bind(on_release=self.e_det_dropdown.open)
        self.e_det_dropdown.bind(on_select=lambda ins, x: setattr(self.type_button_2, 'text', x))

        self.rank_label = Label(text="Enter rank:", size_hint=(0.5, 0.5))
        self.ratio_label = Label(text="Enter ratio:", size_hint=(0.5, 0.5))

        self.rank_input = TextInput(multiline=False, size_hint=(0.4, 0.4))
        self.rank_input.bind(text=self.rank_input_f)
        self.ratio_input = TextInput(multiline=False, size_hint=(0.4, 0.4))
        self.ratio_input.bind(text=self.ratio_input_f)

        self.file_chooser = FileChooserIconView(on_submit=self.submit_file)

        self.mode = None
        self.svd_rank = None
        self.svd_rgb = None
        self.hw_rgb = None
        self.hw_normalization = None
        self.hw_ratio = None
        self.conv_mode = None
        self.blur_mode = None
        self.e_det_mode = None
        self.num = 0

    def on_svd(self, ins):
        self.second_in_layout.clear_widgets()
        self.second_in_layout.add_widget(self.svd_rank_layout)
        self.second_in_layout.add_widget(self.svd_number_layout)
        self.second_in_layout.add_widget(self.rgb_layout_1)
        self.mode = 0

    def on_hw(self, ins):
        self.second_in_layout.clear_widgets()
        self.second_in_layout.add_widget(self.hw_ratio_layout)
        self.second_in_layout.add_widget(self.hw_number_layout)
        self.second_in_layout.add_widget(self.hw_normalization_layout)
        self.second_in_layout.add_widget(self.rgb_layout_2)
        self.mode = 1

    def on_conv(self, ins):
        self.second_in_layout.clear_widgets()
        self.second_in_layout.add_widget(self.conv_choice_layout)
        self.second_in_layout.add_widget(self.conv_dd_layout)
        self.mode = 2

    def on_blur(self, ins):
        self.conv_dd_layout.clear_widgets()
        self.conv_dd_layout.add_widget(self.type_button_1)
        self.conv_mode = 0

    def on_sharpen(self, ins):
        self.conv_dd_layout.clear_widgets()
        self.conv_mode = 1

    def on_e_det(self, ins):
        self.conv_dd_layout.clear_widgets()
        self.conv_dd_layout.add_widget(self.type_button_2)
        self.conv_mode = 2

    def reset(self, ins):
        self.third_in_layout.clear_widgets()
        self.third_in_layout.add_widget(self.file_chooser)

    def normalization_true(self, ins):
        self.hw_normalization = True

    def normalization_false(self, ins):
        self.hw_normalization = False

    def hw_rgb_true(self, ins):
        self.hw_rgb = True

    def hw_rgb_false(self, ins):
        self.hw_rgb = False

    def svd_rgb_true(self, ins):
        self.svd_rgb = True

    def svd_rgb_false(self, ins):
        self.svd_rgb = False

    def blur_true(self, ins):
        self.blur_mode = True

    def blur_false(self, ins):
        self.blur_mode = False

    def e_det_true(self, ins):
        self.e_det_mode = True

    def e_det_false(self, ins):
        self.e_det_mode = False

    def rank_input_f(self, ins, text):
        try:
            self.svd_rank = int(text)
        except ValueError:
            self.svd_rank = None

    def ratio_input_f(self, ins, text):
        try:
            self.hw_ratio = float(text)
        except ValueError:
            self.hw_ratio = None

    def submit_file(self, ins, filepath, touch):
        try:
            filepath = filepath[0]
        except IndexError():
            return None

        if filepath.endswith(".png") or filepath.endswith(".jpg"):
            if self.mode == 0:
                if None in (self.svd_rank, self.svd_rgb):
                    return None
            elif self.mode == 1:
                if None in (self.hw_ratio, self.hw_rgb, self.hw_normalization):
                    return None
            elif self.mode == 2:
                if self.conv_mode is None:
                    return None
                elif self.conv_mode == 0:
                    if self.blur_mode is None:
                        return None
                elif self.conv_mode == 2:
                    if self.e_det_mode is None:
                        return None
            else:
                return None

            if self.mode == 0:
                try:
                    self.num += 1
                    rank_approx.approx(filepath, f"output{self.num}.png", self.svd_rank, rgb=self.svd_rgb)
                    image = Image(source=f"output{self.num}.png")
                    self.image_layout.clear_widgets()
                    self.third_in_layout.clear_widgets()
                    self.image_layout.add_widget(image)
                    self.third_in_layout.add_widget(self.image_layout)
                    self.third_in_layout.add_widget(self.reset_layout)
                except Exception():
                    self.third_in_layout.clear_widgets()
                    self.third_in_layout.add_widget(self.file_chooser)
            elif self.mode == 1:
                try:
                    self.num += 1
                    haar_wavelet.compress(filepath, f"output{self.num}.png", ratio=self.hw_ratio, rgb=self.hw_rgb, normalization=self.hw_normalization)
                    image = Image(source=f"output{self.num}.png")
                    self.image_layout.clear_widgets()
                    self.third_in_layout.clear_widgets()
                    self.image_layout.add_widget(image)
                    self.third_in_layout.add_widget(self.image_layout)
                    self.third_in_layout.add_widget(self.reset_layout)
                except Exception():
                    self.third_in_layout.clear_widgets()
                    self.third_in_layout.add_widget(self.file_chooser)
            elif self.mode == 2:
                try:
                    self.num += 1
                    if self.conv_mode == 0:
                        image_convolution.blur(filepath, self.blur_mode, f"output{self.num}.png")
                    elif self.conv_mode == 1:
                        image_convolution.sharpen(filepath, f"output{self.num}.png")
                    elif self.conv_mode == 2:
                        image_convolution.edge_detection(filepath, self.e_det_mode, f"output{self.num}.png")
                    image = Image(source=f"output{self.num}.png")
                    self.image_layout.clear_widgets()
                    self.third_in_layout.clear_widgets()
                    self.image_layout.add_widget(image)
                    self.third_in_layout.add_widget(self.image_layout)
                    self.third_in_layout.add_widget(self.reset_layout)
                except Exception():
                    self.third_in_layout.clear_widgets()
                    self.third_in_layout.add_widget(self.file_chooser)

    def build(self):
        self.first_in_layout.add_widget(self.svd_button)
        self.first_in_layout.add_widget(self.haar_wavelet_button)
        self.first_in_layout.add_widget(self.convolution_button)
        self.third_in_layout.add_widget(self.file_chooser)

        self.outer_layout.add_widget(self.first_in_layout)
        self.outer_layout.add_widget(self.second_in_layout)
        self.outer_layout.add_widget(self.third_in_layout)

        self.svd_rank_layout.add_widget(self.rank_label)
        self.svd_number_layout.add_widget(self.rank_input)

        self.hw_ratio_layout.add_widget(self.ratio_label)
        self.hw_number_layout.add_widget(self.ratio_input)
        self.hw_normalization_layout.add_widget(self.normalization_button)
        self.rgb_layout_1.add_widget(self.rgb_button_1)
        self.rgb_layout_2.add_widget(self.rgb_button_2)

        self.conv_choice_layout.add_widget(self.blur_button)
        self.conv_choice_layout.add_widget(self.sharpen_button)
        self.conv_choice_layout.add_widget(self.e_det_button)

        self.reset_layout.add_widget(self.reset_button)

        return self.outer_layout


if __name__ == "__main__":
    ImageApp().run()
