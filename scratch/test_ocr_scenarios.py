import os
import io
import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://127.0.0.1:8000"
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
os.makedirs(TEST_DIR, exist_ok=True)

results = {}

print("=== STARTING 10 OCR INTEGRATION TEST SCENARIOS ===\n")

# -------------------------------------------------------------
# Test 1: Image containing clearly readable text
# -------------------------------------------------------------
img1 = Image.new("RGB", (400, 150), color=(255, 255, 255))
d1 = ImageDraw.Draw(img1)
d1.text((30, 50), "STOP RACISM NOW", fill=(0, 0, 0))
buf1 = io.BytesIO()
img1.save(buf1, format="JPEG")
bytes1 = buf1.getvalue()

res1 = requests.post(f"{BASE_URL}/api/extract-text", files={"image": ("test1.jpg", bytes1, "image/jpeg")})
data1 = res1.json()
print("Scenario 1 (Clearly Readable Text):", res1.status_code, data1)
assert res1.status_code == 200 and data1["text_detected"] is True
assert "RACISM" in data1["extracted_text"].upper() or "STOP" in data1["extracted_text"].upper()
results["Test 1: Clearly Readable Text"] = "PASSED"

# -------------------------------------------------------------
# Test 2: Image containing multiple lines of text
# -------------------------------------------------------------
img2 = Image.new("RGB", (500, 250), color=(240, 240, 240))
d2 = ImageDraw.Draw(img2)
d2.text((40, 40), "WHEN YOU TRY YOUR BEST", fill=(10, 10, 80))
d2.text((40, 90), "BUT YOU DO NOT SUCCEED", fill=(10, 10, 80))
d2.text((40, 140), "INTERNET MEMES 2026", fill=(10, 10, 80))
buf2 = io.BytesIO()
img2.save(buf2, format="JPEG")
bytes2 = buf2.getvalue()

res2 = requests.post(f"{BASE_URL}/api/extract-text", files={"image": ("test2.jpg", bytes2, "image/jpeg")})
data2 = res2.json()
print("Scenario 2 (Multiple Lines Text):", res2.status_code, data2)
assert res2.status_code == 200 and data2["text_detected"] is True
assert len(data2["raw_lines"]) >= 2
results["Test 2: Multiple Lines of Text"] = "PASSED"

# -------------------------------------------------------------
# Test 3: Image with no text
# -------------------------------------------------------------
img3 = Image.new("RGB", (300, 300), color=(120, 150, 180))
buf3 = io.BytesIO()
img3.save(buf3, format="PNG")
bytes3 = buf3.getvalue()

res3 = requests.post(f"{BASE_URL}/api/extract-text", files={"image": ("test3.png", bytes3, "image/png")})
data3 = res3.json()
print("Scenario 3 (No Text Image):", res3.status_code, data3)
assert res3.status_code == 200
assert data3["text_detected"] is False and data3["extracted_text"] == ""
results["Test 3: Image with No Text"] = "PASSED"

# -------------------------------------------------------------
# Test 4: Image with difficult-to-read text (noisy background)
# -------------------------------------------------------------
img4 = Image.new("RGB", (400, 200), color=(80, 80, 80))
d4 = ImageDraw.Draw(img4)
for i in range(0, 400, 10):
    d4.line([(i, 0), (i, 200)], fill=(70, 70, 70))
d4.text((30, 80), "SUBTLE LOW CONTRAST", fill=(95, 95, 95))
buf4 = io.BytesIO()
img4.save(buf4, format="JPEG")
bytes4 = buf4.getvalue()

res4 = requests.post(f"{BASE_URL}/api/extract-text", files={"image": ("test4.jpg", bytes4, "image/jpeg")})
data4 = res4.json()
print("Scenario 4 (Difficult to Read):", res4.status_code, data4)
assert res4.status_code == 200  # API handles low contrast gracefully without crashing
results["Test 4: Difficult-to-Read Text"] = "PASSED"

# -------------------------------------------------------------
# Test 5: Unsupported file format (.txt)
# -------------------------------------------------------------
res5 = requests.post(
    f"{BASE_URL}/api/extract-text",
    files={"image": ("invalid.txt", b"plain text file contents", "text/plain")}
)
print("Scenario 5 (Unsupported File):", res5.status_code, res5.json())
assert res5.status_code == 400
assert "Unsupported file format" in res5.json()["detail"]
results["Test 5: Unsupported File (.txt)"] = "PASSED"

# -------------------------------------------------------------
# Test 6: Corrupted image file
# -------------------------------------------------------------
res6 = requests.post(
    f"{BASE_URL}/api/extract-text",
    files={"image": ("corrupted.jpg", b"RIFF\x00\x00\x00GARBAGEHEADERNOTJPEG", "image/jpeg")}
)
print("Scenario 6 (Corrupted Image):", res6.status_code, res6.json())
assert res6.status_code == 400
assert "Invalid image" in res6.json()["detail"] or "corrupted" in res6.json()["detail"]
results["Test 6: Corrupted Image File"] = "PASSED"

# -------------------------------------------------------------
# Test 7: User editing OCR output before analysis
# -------------------------------------------------------------
# User uploads img1 -> OCR returns data1["extracted_text"]
ocr_text = data1["extracted_text"]
# User corrects or edits typo:
user_corrected_text = ocr_text + " - corrected by moderator"
print(f"Scenario 7 (Edited Text Flow): OCR='{ocr_text}' -> UserEdited='{user_corrected_text}'")
res7 = requests.post(
    f"{BASE_URL}/api/predict",
    files={"image": ("test1.jpg", bytes1, "image/jpeg")},
    data={"text": user_corrected_text}
)
data7 = res7.json()
print("Scenario 7 Prediction Result with Edited Text:", res7.status_code, data7["label"], data7["confidence"])
assert res7.status_code == 200 and data7["success"] is True
results["Test 7: User Editing OCR Output"] = "PASSED"

# -------------------------------------------------------------
# Test 8: OCR output passed successfully to BERT tokenizer pipeline
# -------------------------------------------------------------
# Pass the raw OCR output directly to /api/predict
res8 = requests.post(
    f"{BASE_URL}/api/predict",
    files={"image": ("test2.jpg", bytes2, "image/jpeg")},
    data={"text": data2["extracted_text"]}
)
data8 = res8.json()
print("Scenario 8 (Direct OCR -> BERT):", res8.status_code, data8["model_name"])
assert res8.status_code == 200 and data8["success"] is True
results["Test 8: OCR Output to BERT Pipeline"] = "PASSED"

# -------------------------------------------------------------
# Test 9: Complete Multimodal Inference with ResNet50 + BERT + BAM
# -------------------------------------------------------------
# Verify attention weights, class probabilities, and calibrated threshold tau*
assert "attention" in data8 and "visual" in data8["attention"] and "textual" in data8["attention"]
assert "calibrated_threshold" in data8 and data8["calibrated_threshold"] == 0.41
assert abs((data8["probability_hateful"] + data8["probability_non_hateful"]) - 100.0) < 0.5
print(f"Scenario 9 (Full BAM Prediction): Class={data8['predicted_class']} ({data8['label']}), Visual={data8['attention']['visual']}%, Textual={data8['attention']['textual']}%")
results["Test 9: Full Multimodal BAM Inference"] = "PASSED"

# -------------------------------------------------------------
# Test 10: Empty image / API Failure Handling
# -------------------------------------------------------------
res10 = requests.post(
    f"{BASE_URL}/api/extract-text",
    files={"image": ("empty.jpg", b"", "image/jpeg")}
)
print("Scenario 10 (Empty Image Handling):", res10.status_code, res10.json())
assert res10.status_code == 400
assert "empty" in res10.json()["detail"].lower()
results["Test 10: OCR/API Failure Handling"] = "PASSED"

print("\n=== SUMMARY OF ALL 10 TESTS ===")
for test_name, status in results.items():
    print(f"  {test_name}: {status}")

assert len(results) == 10
print("\nALL 10 TEST SCENARIOS COMPLETED AND VALIDATED SUCCESSFULLY!")
