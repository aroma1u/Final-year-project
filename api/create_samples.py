import os
from PIL import Image, ImageDraw, ImageFont

samples_dir = os.path.join("api", "static", "samples")
os.makedirs(samples_dir, exist_ok=True)

# 1. Create Sample Benign Image (Calm landscape with subtle gradient)
img_benign = Image.new("RGB", (600, 400), color=(70, 130, 180))
draw = ImageDraw.Draw(img_benign)
# Draw nice geometric patterns
for i in range(10):
    draw.rectangle([i*20, i*15, 600 - i*20, 400 - i*15], outline=(100 + i*10, 160 + i*8, 210), width=2)
draw.text((150, 190), "[MemeGuard AI Sample A: Diversity Meme]", fill=(255, 255, 255))
img_benign.save(os.path.join(samples_dir, "sample_benign.jpg"), quality=90)

# 2. Create Sample Irony Image (Contrasting bold meme graphic)
img_irony = Image.new("RGB", (600, 400), color=(180, 80, 70))
draw = ImageDraw.Draw(img_irony)
for i in range(10):
    draw.ellipse([i*25, i*15, 600 - i*25, 400 - i*15], outline=(220, 110 + i*10, 100), width=2)
draw.text((140, 190), "[MemeGuard AI Sample B: Complex Juxtaposition]", fill=(255, 255, 255))
img_irony.save(os.path.join(samples_dir, "sample_irony.jpg"), quality=90)

print(f"Generated sample demonstration images in '{samples_dir}'.")
