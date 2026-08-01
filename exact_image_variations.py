from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import os
from datetime import datetime
import random

class ExactImageVariations:
    def __init__(self):
        self.output_folder = "exact_car_variations"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_image(self, image_path):
        """Load image with error handling"""
        try:
            image = Image.open(image_path)
            return image.convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def save_variation(self, image, variation_name):
        """Save image variation with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_folder}/{variation_name}_{timestamp}.jpg"
        image.save(filename, quality=95)
        return filename
    
    def color_variations(self, image):
        """Create different color variations of the exact same car"""
        variations = []
        
        # Original (for reference)
        variations.append(("original", image))
        
        # Color enhancements
        enhancer = ImageEnhance.Color(image)
        variations.append(("enhanced_colors", enhancer.enhance(1.5)))
        variations.append(("desaturated", enhancer.enhance(0.5)))
        
        # Brightness variations
        brightness_enhancer = ImageEnhance.Brightness(image)
        variations.append(("brighter", brightness_enhancer.enhance(1.3)))
        variations.append(("darker", brightness_enhancer.enhance(0.7)))
        
        # Contrast variations
        contrast_enhancer = ImageEnhance.Contrast(image)
        variations.append(("high_contrast", contrast_enhancer.enhance(1.4)))
        variations.append(("low_contrast", contrast_enhancer.enhance(0.6)))
        
        # Color temperature variations (warm/cool)
        np_image = np.array(image)
        
        # Warm tone (add red/yellow)
        warm_image = np_image.copy().astype(np.float32)
        warm_image[:,:,0] = np.minimum(warm_image[:,:,0] * 1.2, 255)  # Red
        warm_image[:,:,1] = np.minimum(warm_image[:,:,1] * 1.1, 255)  # Green
        variations.append(("warm_tone", Image.fromarray(warm_image.astype(np.uint8))))
        
        # Cool tone (add blue)
        cool_image = np_image.copy().astype(np.float32)
        cool_image[:,:,2] = np.minimum(cool_image[:,:,2] * 1.2, 255)  # Blue
        variations.append(("cool_tone", Image.fromarray(cool_image.astype(np.uint8))))
        
        return variations
    
    def background_variations(self, image):
        """Keep car exactly the same but change background"""
        variations = []
        
        # Convert to numpy for processing
        np_image = np.array(image)
        
        # Simple background removal (assuming car is the main subject)
        # This is a simplified approach - for best results use proper segmentation
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        
        # Create mask (this is basic - you might need better segmentation)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.medianBlur(mask, 5)
        
        # Different backgrounds
        backgrounds = [
            ("urban_background", self.create_urban_background(image.size)),
            ("nature_background", self.create_nature_background(image.size)),
            ("showroom_background", self.create_showroom_background(image.size)),
            ("gradient_background", self.create_gradient_background(image.size)),
        ]
        
        for bg_name, background in backgrounds:
            # Composite car onto new background
            background_np = np.array(background)
            result = np_image.copy()
            
            # Use mask to blend
            for i in range(3):
                result[:,:,i] = np.where(mask > 128, np_image[:,:,i], background_np[:,:,i])
            
            variations.append((bg_name, Image.fromarray(result)))
        
        return variations
    
    def create_urban_background(self, size):
        """Create urban street background"""
        bg = Image.new('RGB', size, '#2C3E50')
        return bg
    
    def create_nature_background(self, size):
        """Create natural scenery background"""
        bg = Image.new('RGB', size, '#27AE60')
        # Add some simple gradient
        for y in range(size[1]):
            color_val = int(100 + (155 * y / size[1]))
            bg.paste((0, color_val, 0), (0, y, size[0], y+1))
        return bg
    
    def create_showroom_background(self, size):
        """Create professional showroom background"""
        bg = Image.new('RGB', size, '#ECF0F1')
        # Add subtle gradient
        for y in range(size[1]):
            color_val = int(220 + (35 * y / size[1]))
            bg.paste((color_val, color_val, color_val), (0, y, size[0], y+1))
        return bg
    
    def create_gradient_background(self, size):
        """Create colorful gradient background"""
        bg = Image.new('RGB', size, (0, 0, 0))
        for y in range(size[1]):
            r = int(255 * y / size[1])
            g = int(128 * y / size[1])
            b = int(255 * (1 - y / size[1]))
            bg.paste((r, g, b), (0, y, size[0], y+1))
        return bg
    
    def lighting_variations(self, image):
        """Simulate different lighting conditions"""
        variations = []
        
        # Original
        variations.append(("original_lighting", image))
        
        # Convert to numpy for processing
        np_image = np.array(image).astype(np.float32)
        
        # Golden hour (warm, soft light)
        golden_hour = np_image.copy()
        golden_hour[:,:,0] *= 1.3  # More red
        golden_hour[:,:,1] *= 1.1  # More green
        golden_hour = np.clip(golden_hour, 0, 255)
        variations.append(("golden_hour", Image.fromarray(golden_hour.astype(np.uint8))))
        
        # Blue hour (cool, evening light)
        blue_hour = np_image.copy()
        blue_hour[:,:,2] *= 1.3  # More blue
        blue_hour[:,:,0] *= 0.9  # Less red
        blue_hour = np.clip(blue_hour, 0, 255)
        variations.append(("blue_hour", Image.fromarray(blue_hour.astype(np.uint8))))
        
        # Overcast (soft, flat light)
        overcast = np_image.copy() * 0.8  # Reduce overall brightness
        overcast = np.clip(overcast, 0, 255)
        variations.append(("overcast", Image.fromarray(overcast.astype(np.uint8))))
        
        # Studio lighting (high contrast)
        studio = np_image.copy()
        studio = cv2.convertScaleAbs(studio, alpha=1.2, beta=10)  # Increase contrast
        variations.append(("studio_lighting", Image.fromarray(studio)))
        
        return variations
    
    def artistic_variations(self, image):
        """Apply artistic filters while keeping car recognizable"""
        variations = []
        
        # Original
        variations.append(("original", image))
        
        # Sharpened
        sharpened = image.filter(ImageFilter.SHARPEN)
        variations.append(("sharpened", sharpened))
        
        # Soft focus
        soft = image.filter(ImageFilter.GaussianBlur(1))
        variations.append(("soft_focus", soft))
        
        # Vintage effect
        vintage = ImageOps.colorize(image.convert('L'), '#704214', '#F0E68C')
        variations.append(("vintage", vintage))
        
        # Black and white
        bw = image.convert('L').convert('RGB')
        variations.append(("black_white", bw))
        
        # Sepia
        sepia = image.convert('L')
        sepia = ImageOps.colorize(sepia, '#704214', '#F0E68C')
        variations.append(("sepia", sepia))
        
        return variations
    
    def perspective_variations(self, image):
        """Create different perspective views (simulated)"""
        variations = []
        
        # Original
        variations.append(("original_perspective", image))
        
        # Convert to numpy for OpenCV processing
        np_image = np.array(image)
        height, width = np_image.shape[:2]
        
        # Slight rotation variations
        for angle, name in [(-5, "rotated_left"), (5, "rotated_right")]:
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(np_image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            variations.append((name, Image.fromarray(rotated)))
        
        # Perspective transform (simulate different camera angles)
        pts1 = np.float32([[0,0], [width,0], [0,height], [width,height]])
        pts2 = np.float32([[10,10], [width-10,20], [10,height-10], [width-10,height-20]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective = cv2.warpPerspective(np_image, matrix, (width, height))
        variations.append(("low_angle", Image.fromarray(perspective)))
        
        return variations
    
    def generate_all_variations(self, image_path):
        """Generate all types of variations"""
        original_image = self.load_image(image_path)
        if original_image is None:
            return []
        
        all_variations = []
        
        print("🎨 Generating color variations...")
        all_variations.extend(self.color_variations(original_image))
        
        print("🏞️  Generating background variations...")
        all_variations.extend(self.background_variations(original_image))
        
        print("💡 Generating lighting variations...")
        all_variations.extend(self.lighting_variations(original_image))
        
        print("🖼️  Generating artistic variations...")
        all_variations.extend(self.artistic_variations(original_image))
        
        print("📐 Generating perspective variations...")
        all_variations.extend(self.perspective_variations(original_image))
        
        # Save all variations
        saved_files = []
        for variation_name, variation_image in all_variations:
            filename = self.save_variation(variation_image, variation_name)
            saved_files.append((variation_name, filename))
            print(f"✓ Created: {variation_name}")
        
        return saved_files

def main():
    generator = ExactImageVariations()
    
    print("🚗 Exact Image Variation Generator")
    print("📸 Creating pixel-perfect car variations...")
    
    # Replace with your actual car image path
    image_path = "your_car_image.jpg"  # CHANGE THIS TO YOUR IMAGE PATH
    
    if os.path.exists(image_path):
        variations = generator.generate_all_variations(image_path)
        print(f"\n✨ Successfully created {len(variations)} exact image variations!")
        print(f"📁 Location: {os.path.abspath(generator.output_folder)}")
        
        # Show some examples
        print("\n🎯 Sample variations created:")
        for i, (name, path) in enumerate(variations[:10]):  # Show first 10
            print(f"  {i+1}. {name}: {os.path.basename(path)}")
            
    else:
        print(f"❌ Image not found: {image_path}")
        print("💡 Please update the 'image_path' variable with your actual image path")

if __name__ == "__main__":
    main()