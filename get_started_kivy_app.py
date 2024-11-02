from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.garden.xcamera import XCamera  
from kivy.utils import platform
from plyer import filechooser
from ultralytics import YOLO
import cv2

class MainApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        # Set the background color to blue
        self.layout.background_color = (0, 0, 1, 1) 

        # Image Display
        self.image = Image(source='project_banner.jpg')  
        self.layout.add_widget(self.image)

        # Buttons
        button_layout = BoxLayout(size_hint_y=0.1)

        load_button = Button(text="Load Image")
        load_button.bind(on_press=self.load_image)
        button_layout.add_widget(load_button)

        if platform == 'android':
            take_picture_button = Button(text="Take Picture")
            take_picture_button.bind(on_press=self.take_picture)
            button_layout.add_widget(take_picture_button)

        # Create the dropdown outside the button_layout
        self.dropdown = DropDown() 
        for crop in ["Maize", "Beans", "Cassava"]:
            btn = Button(text=crop, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)

        self.dropdown_button = Button(text='Select Crop')
        self.dropdown_button.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self.dropdown_button, 'text', x))
        button_layout.add_widget(self.dropdown_button)

        inspect_button = Button(text="Inspect")
        inspect_button.bind(on_press=self.inspect_image)
        button_layout.add_widget(inspect_button)

        self.layout.add_widget(button_layout)

        # Placeholder for inspection results 
        self.result_label = Label(text="", size_hint_y=0.1)
        self.layout.add_widget(self.result_label)

        return self.layout

    def load_image(self, instance):
        filechooser.open_file(on_selection=self.handle_image_selection)

    def handle_image_selection(self, selection):
        if selection:
            self.image.source = selection[0]
            self.image.reload()

    def take_picture(self, instance):
        cam = XCamera(on_picture_taken=self.handle_picture_taken)
        cam.play = True

    def handle_picture_taken(self, camera, filename):
        self.image.source = filename
        self.image.reload()
        camera.play = False

    def inspect_image(self, instance):
        crop_type = self.dropdown_button.text
        image_path = self.image.source 

        if image_path == 'placeholder.jpg':
            self.result_label.text = "Please load an image first!"
            return

        try:
            predicted_label = self.cropPrediction(cv2.imread(image_path), crop_type.lower())
            self.result_label.text = f"Predicted Label: {predicted_label}"
        except Exception as e:
            self.result_label.text = f"Error: {e}"

    def cropPrediction(self, input_image, crop_to_inspect):
        """ 
        Function for predicting crop disease
        Input Parameters:
        input_image --> image path of the image to be predicted --> str
        crop_to_inspect --> the selected crop from the dropdown --> str
        Output:
        predicted_label --> returning the crop disease --> str
        """
        # Predict with the model
        model_paths = {
            "maize": "./prediction_models/maize_best_model.pt",
            "cassava": "./prediction_models/cassava_best_model.pt",
            "beans": "./prediction_models/beans_best_model.pt"
        }

        if crop_to_inspect in model_paths:
            model = YOLO(model_paths[crop_to_inspect])
        else:
            return "Please select a valid crop type."

        results = model.predict(input_image, device="cpu", save=False, verbose=False)
        labels_names = results[0].names
        pred_cls_idx = results[0].probs.top1
        predicted_label = labels_names[pred_cls_idx]
        print(predicted_label)
        return predicted_label

if __name__ == '__main__':
    MainApp().run()