import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import os
from datetime import datetime

class ImageTo3DCarGenerator:
    def __init__(self):
        self.output_folder = "3d_from_images"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def extract_car_silhouette(self, image_path):
        """Extract car silhouette from image for 3D modeling"""
        image = Image.open(image_path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Use multiple methods to detect car
        # Method 1: Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Method 2: Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine methods
        combined = cv2.bitwise_or(edges, thresh)
        
        # Find largest contour (assuming it's the car)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask from largest contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            return mask, largest_contour
        return None, None
    
    def create_3d_from_silhouette(self, silhouette, contour, car_type="sedan"):
        """Create 3D car model from silhouette"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get bounding box of car
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create 3D model based on silhouette
        # Scale factors for different car types
        if car_type == "sports":
            length_scale, width_scale, height_scale = 1.2, 0.9, 0.8
        elif car_type == "suv":
            length_scale, width_scale, height_scale = 1.0, 1.1, 1.2
        else:  # sedan
            length_scale, width_scale, height_scale = 1.0, 1.0, 1.0
        
        length = w * length_scale
        width = h * width_scale * 0.3  # Car is narrower than silhouette
        height = h * height_scale * 0.2
        
        # Create car body
        x_vals = np.linspace(-length/2, length/2, 20)
        y_vals = np.linspace(-width/2, width/2, 15)
        z_vals = np.linspace(0, height, 10)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Car body shape (simplified)
        Z_body = np.zeros_like(X)
        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                xi, yj = x_vals[i], y_vals[j]
                # Curved roof effect
                roof_height = height * (1 - (xi/(length/2))**4) * (1 - (yj/(width/2))**4)
                Z_body[j,i] = roof_height
        
        ax.plot_surface(X, Y, Z_body, alpha=0.8, color='blue')
        
        # Add wheels based on silhouette proportions
        wheel_radius = height * 0.3
        wheel_positions = [
            (-length/3, -width/2), (-length/3, width/2),
            (length/3, -width/2), (length/3, width/2)
        ]
        
        for wx, wy in wheel_positions:
            theta = np.linspace(0, 2*np.pi, 20)
            z_wheel = np.linspace(-wheel_radius/3, wheel_radius/3, 5)
            Theta, Z_wheel = np.meshgrid(theta, z_wheel)
            
            X_wheel = wx + wheel_radius * np.cos(Theta)
            Y_wheel = wy + Z_wheel
            Z_wheel = wheel_radius * np.sin(Theta)
            
            ax.plot_surface(X_wheel, Y_wheel, Z_wheel, color='black', alpha=0.9)
        
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_zlabel('Height')
        ax.set_title(f'3D {car_type.capitalize()} from Image')
        ax.set_box_aspect([1, 1, 0.5])
        
        return fig
    
    def generate_3d_variations(self, image_path):
        """Generate multiple 3D variations from a single image"""
        silhouette, contour = self.extract_car_silhouette(image_path)
        if silhouette is None:
            print("❌ Could not extract car silhouette")
            return []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = []
        
        car_types = ["sedan", "sports", "suv"]
        
        for car_type in car_types:
            fig = self.create_3d_from_silhouette(silhouette, contour, car_type)
            
            filename = f"{self.output_folder}/3d_{car_type}_from_image_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            generated_files.append(filename)
            print(f"✓ Created 3D {car_type}: {filename}")
        
        return generated_files

def main():
    generator = ImageTo3DCarGenerator()
    
    print("🚗 Image to 3D Car Generator")
    print("📐 Creating 3D models from car images...")
    
    # Replace with your image path
    sample_image_path = "sample_car.jpg"
    
    if os.path.exists(sample_image_path):
        files = generator.generate_3d_variations(sample_image_path)
        print(f"\n✨ Created {len(files)} 3D car variations!")
    else:
        print("⚠️  Please add a car image named 'sample_car.jpg' to test")

if __name__ == "__main__":
    main()