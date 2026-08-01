from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import os
from datetime import datetime
import svgwrite

class ImageToSVGTracer:
    def __init__(self):
        self.output_folder = "traced_cars"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for tracing"""
        try:
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def trace_image_to_svg(self, image_path, output_svg_path, simplify=True):
        """Trace a car image and convert to SVG"""
        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None
        
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create SVG
        height, width = gray.shape
        dwg = svgwrite.Drawing(output_svg_path, size=(f"{width}px", f"{height}px"))
        
        # Add original image as background (optional)
        # dwg.add(dwg.image(href=image_path, insert=(0, 0), size=(f"{width}px", f"{height}px")))
        
        # Draw contours as SVG paths
        for contour in contours:
            if len(contour) > 2:  # Need at least 3 points for a polygon
                if simplify:
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(contour) > 2:
                    points = [(float(point[0][0]), float(point[0][1])) for point in contour]
                    dwg.add(dwg.polygon(points=points, fill='none', stroke='black', stroke_width=1))
        
        dwg.save()
        return dwg
    
    def create_car_variations_from_image(self, image_path, num_variations=3):
        """Create multiple car variations from a single image"""
        original_image = self.load_and_preprocess_image(image_path)
        if original_image is None:
            return []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = []
        
        for i in range(num_variations):
            # Create different traced versions
            output_path = f"{self.output_folder}/traced_car_{timestamp}_v{i+1}.svg"
            
            # Vary the edge detection parameters for different styles
            canny_low = 30 + i * 20
            canny_high = 100 + i * 30
            
            # Trace with different parameters
            self.trace_with_varied_parameters(image_path, output_path, canny_low, canny_high)
            generated_files.append(output_path)
            print(f"✓ Created variation {i+1}: {output_path}")
        
        return generated_files
    
    def trace_with_varied_parameters(self, image_path, output_path, canny_low, canny_high):
        """Trace image with specific parameters"""
        image = self.load_and_preprocess_image(image_path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray.shape
        dwg = svgwrite.Drawing(output_path, size=(f"{width}px", f"{height}px"))
        
        for contour in contours:
            if len(contour) > 2:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(contour) > 2:
                    points = [(float(point[0][0]), float(point[0][1])) for point in contour]
                    dwg.add(dwg.polygon(points=points, fill='none', stroke='black', stroke_width=1))
        
        dwg.save()
    
    def generate_stylized_car_from_image(self, image_path, style="cartoon"):
        """Generate stylized car SVG from image"""
        image = self.load_and_preprocess_image(image_path)
        if image is None:
            return None
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Different processing based on style
        if style == "cartoon":
            # Cartoon effect
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 9)
            
            # Color quantization for cartoon effect
            data = cv_image.reshape((-1, 3))
            data = np.float32(data)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            cartoon = centers[labels.flatten()].reshape(cv_image.shape)
            
        elif style == "silhouette":
            # Silhouette effect
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, edges = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Convert to SVG
        height, width = edges.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_folder}/stylized_{style}_car_{timestamp}.svg"
        
        dwg = svgwrite.Drawing(output_path, size=(f"{width}px", f"{height}px"))
        
        if style == "cartoon":
            # For cartoon, we'll use the edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 2:
                    points = [(float(point[0][0]), float(point[0][1])) for point in contour]
                    dwg.add(dwg.polygon(points=points, fill='black', stroke='none'))
        
        dwg.save()
        return output_path

def main():
    tracer = ImageToSVGTracer()
    
    print("🎨 Image to SVG Car Tracer")
    print("📁 Place your car images in the same folder")
    print("🔄 Generating variations...")
    
    # Example usage - replace with your image path
    sample_image_path = "sample_car.jpg"  # Change this to your image path
    
    if os.path.exists(sample_image_path):
        # Create traced variations
        variations = tracer.create_car_variations_from_image(sample_image_path, 3)
        
        # Create stylized versions
        styles = ["cartoon", "silhouette"]
        for style in styles:
            output = tracer.generate_stylized_car_from_image(sample_image_path, style)
            if output:
                print(f"✓ Created {style} version: {output}")
    
    else:
        print("⚠️  Please add a car image named 'sample_car.jpg' to test")
        # Create a sample demonstration
        print("🔄 Creating sample demonstration...")
        # You would add code here to create a sample car image for testing

if __name__ == "__main__":
    main()