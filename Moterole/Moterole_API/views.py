import numpy as np
import tensorflow as tf
import cv2
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)

class PredictView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = os.path.join(os.path.dirname(__file__), 'MoteroleCNNModel.keras')
        logging.info(f"Loading model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except OSError as e:
            raise RuntimeError("Failed to load the model. Check the model path.") from e

        self.char_map = {
            # Map indices to characters here...
        }

    def post(self, request):
        user_image = request.data.get('image')
        if not user_image:
            return Response({"error": "No image provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            img_data = base64.b64decode(user_image)
            np_arr = np.frombuffer(img_data, np.uint8)
            user_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            return Response({"error": "Could not process image."}, status=status.HTTP_400_BAD_REQUEST)

        user_image_resized = cv2.resize(user_image, (28, 28))
        user_image_norm = user_image_resized.astype('float32') / 255.0
        user_image_norm = user_image_norm.reshape(1, 28, 28, 1)

        # Detect Lines and Shapes
        line_detection_result = self.detect_lines(user_image)
        shape_detection_result = self.detect_shapes(user_image)

        # Make a letter prediction
        prediction = self.model.predict(user_image_norm)
        predicted_class = np.argmax(prediction, axis=1).tolist()[0]
        confidence = max(prediction[0]) * 100  # Confidence percentage
        predicted_letter = self.char_map.get(predicted_class, "Unknown")

        return Response({
            'prediction': predicted_letter,
            'confidence': confidence,
            'line_detection': line_detection_result,
            'shape_detection': shape_detection_result
        }, status=status.HTTP_200_OK)

    def detect_lines(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is not None:
            # Logic to classify lines (horizontal, vertical, etc.)
            # Return results as a dictionary with precision percentages
            return {
                'horizontal': 75,  # Replace with actual computation
                'vertical': 85,
                'curve': 0,
                'zigzag': 0,
                'slanting': 0
            }
        return {}

    def detect_shapes(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_results = {
            'triangle': 0,
            'circle': 0,
            'oval': 0,
            'square': 0,
            'rectangle': 0
        }

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 3:
                shape_results['triangle'] += 1
            elif len(approx) == 4:
                aspect_ratio = float(cv2.boundingRect(approx)[2]) / float(cv2.boundingRect(approx)[3])
                shape_results['square' if aspect_ratio == 1 else 'rectangle'] += 1
            elif len(approx) > 4:
                shape_results['circle'] += 1  # Placeholder for circles
            # Add more detailed detection logic for ovals and other shapes

        return shape_results

