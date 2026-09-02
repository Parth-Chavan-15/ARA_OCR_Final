import os
import random
import shutil
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Configuration
BASE_DIR = Path(os.path.abspath(__file__)).parent.parent
DEMO_DIR = BASE_DIR / "data" / "demo"
ORIGINALS_DIR = BASE_DIR / "data" / "originals"

CLASSES = [
    "CASTE_CERTIFICATE",
    "CASTE_VALIDITY_CERTIFICATE",
    "CASTE_VALIDITY_RECEIPT",
    "PROFORMA_O",
    "LEAVING_CERTIFICATE",
    "UNKNOWN_OUT_OF_SCOPE"
]

CANDIDATES = {
    "EN25272514": ["CASTE_CERTIFICATE", "LEAVING_CERTIFICATE"],
    "EN25272515": ["CASTE_VALIDITY_CERTIFICATE", "CASTE_VALIDITY_RECEIPT"],
    "EN25272516": ["PROFORMA_O", "CASTE_CERTIFICATE"],
    "EN25272517": ["LEAVING_CERTIFICATE", "CASTE_VALIDITY_RECEIPT"],
    "EN25272518": ["UNKNOWN_OUT_OF_SCOPE"]
}

OOS_TYPES = ["NCL_Certificate", "Income_Certificate", "Domicile_Certificate"]

def setup_directories():
    for cls in CLASSES:
        os.makedirs(DEMO_DIR / cls, exist_ok=True)
    for cand in CANDIDATES.keys():
        os.makedirs(ORIGINALS_DIR / cand, exist_ok=True)

def get_font(size, bold=False):
    # Try to load Windows fonts, fallback to default
    try:
        font_name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(font_name, size)
    except IOError:
        try:
            font_name = "tahomabd.ttf" if bold else "tahoma.ttf"
            return ImageFont.truetype(font_name, size)
        except IOError:
            return ImageFont.load_default()

def apply_degradations(img):
    choice = random.random()
    if choice < 0.2:
        # Blur
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif choice < 0.4:
        # Rotate slightly
        img = img.rotate(random.uniform(-3, 3), resample=Image.BICUBIC, expand=True, fillcolor="white")
    elif choice < 0.6:
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.6, 1.4))
    elif choice < 0.8:
        # Noise (simple approximation)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.2))
    return img

def create_document(class_name, variation, index):
    width, height = 800, 1100
    img = Image.new('RGB', (width, height), color=(245, 245, 240)) # Off-white
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(36, bold=True)
    header_font = get_font(24, bold=True)
    text_font = get_font(20)
    small_font = get_font(16)
    
    # Border
    draw.rectangle([20, 20, width-20, height-20], outline="black", width=2)
    draw.rectangle([25, 25, width-25, height-25], outline="black", width=1)
    
    # Common seal placeholder
    draw.ellipse([width-150, 50, width-50, 150], outline="blue", width=2)
    draw.text((width-135, 90), "SEAL", fill="blue", font=header_font)
    
    y = 60
    
    if class_name == "CASTE_CERTIFICATE":
        titles = ["CASTE CERTIFICATE", "जातीचा दाखला", "CASTE CERTIFICATE / जातीचा दाखला"]
        title = titles[variation % 3]
        draw.text((width//2, y), title, fill="black", font=title_font, anchor="mt")
        
        content = [
            "Form 6/7/8",
            "Issuing Authority: Tehsildar / Sub-Divisional Officer",
            "Certificate Number: CC/2026/XYZ/789" + str(index),
            "Date: 12/05/2026",
            "--------------------------------------------------",
            "This is to certify that:",
            "Name: Rohan Sharma",
            "Father's Name: Prakash Sharma",
            "Village/Town: Pune",
            "Taluka: Haveli",
            "District: Pune",
            "Belongs to Caste/Tribe: Maratha",
            "Sub-caste: -",
            "Which is recognized as a backward class under:",
            "The Constitution (Scheduled Castes) Order, 1950",
            "The Constitution (Scheduled Tribes) Order, 1950"
        ]
        y += 80
        for line in content:
            draw.text((50, y), line, fill="black", font=text_font)
            y += 40

    elif class_name == "CASTE_VALIDITY_CERTIFICATE":
        titles = ["CERTIFICATE OF VALIDITY OF CASTE CERTIFICATE", "जात वैधता प्रमाणपत्र", "VALIDITY CERTIFICATE / वैधता प्रमाणपत्र"]
        title = titles[variation % 3]
        draw.text((width//2, y), title, fill="black", font=title_font, anchor="mt")
        
        content = [
            "DISTRICT CASTE CERTIFICATE SCRUTINY COMMITTEE, PUNE",
            "Outward No: CVC/2026/889" + str(index),
            "Validity Date: 10/08/2026",
            "--------------------------------------------------",
            "Applicant Name: Rohan Sharma",
            "Caste: Maratha",
            "Caste Certificate No: CC/2026/XYZ/789" + str(index),
            "It is certified that the caste claim of the above applicant",
            "is scrutinized and found VALID.",
            "Seal and Signature of Committee Member"
        ]
        y += 80
        for line in content:
            draw.text((50, y), line, fill="black", font=text_font)
            y += 40

    elif class_name == "CASTE_VALIDITY_RECEIPT":
        title = "ONLINE PAYMENT RECEIPT"
        draw.text((width//2, y), title, fill="black", font=title_font, anchor="mt")
        
        content = [
            "District Caste Certificate Scrutiny Committee",
            "Payment for Caste Validity Application",
            "--------------------------------------------------",
            "Application Number: APP-2026-" + str(1000 + index),
            "Receipt No.: REC-99" + str(index),
            "Transaction Reference: TXN8877" + str(index),
            "Transaction Status: SUCCESS",
            "Amount: Rs. 150.00",
            "Payment Date: 01/08/2026"
        ]
        y += 80
        for line in content:
            draw.text((50, y), line, fill="black", font=text_font)
            y += 40
            
    elif class_name == "PROFORMA_O":
        titles = ["PROFORMA-O", "प्रपत्र-ओ", "PROFORMA-O / प्रपत्र-ओ"]
        title = titles[variation % 3]
        draw.text((width//2, y), title, fill="black", font=title_font, anchor="mt")
        
        content = [
            "Linguistic Minority Certificate",
            "--------------------------------------------------",
            "School Name: Model High School, Pune",
            "This is to certify that:",
            "Student Name: Rohan Sharma",
            "is studying in this institution.",
            "As per the school register, his/her mother tongue is:",
            "Mother Tongue: Marathi",
            "This certificate is issued for admission purpose.",
            "Principal Signature and Seal"
        ]
        y += 80
        for line in content:
            draw.text((50, y), line, fill="black", font=text_font)
            y += 40

    elif class_name == "LEAVING_CERTIFICATE":
        titles = ["LEAVING CERTIFICATE", "शाळा सोडल्याचा दाखला", "LEAVING CERTIFICATE"]
        title = titles[variation % 3]
        draw.text((width//2, y), title, fill="black", font=title_font, anchor="mt")
        
        content = [
            "School Name: Model High School, Pune",
            "UDISE Code: 27252000000",
            "--------------------------------------------------",
            "Student Name: Rohan Sharma",
            "Mother's Name: Sunita Sharma",
            "Nationality: Indian",
            "Mother Tongue: Marathi",
            "Religion: Hindu",
            "Caste: Maratha",
            "Sub-caste: -",
            "Date of Birth: 15/06/2008",
            "Date of Admission: 10/06/2014",
            "Date of Leaving: 31/05/2026",
            "Standard: 12th (HSC)",
            "Conduct: Good"
        ]
        y += 80
        for line in content:
            draw.text((50, y), line, fill="black", font=text_font)
            y += 40

    elif class_name == "UNKNOWN_OUT_OF_SCOPE":
        oos_type = OOS_TYPES[variation % len(OOS_TYPES)]
        title = oos_type.replace("_", " ").upper()
        draw.text((width//2, y), title, fill="black", font=title_font, anchor="mt")
        
        if oos_type == "NCL_Certificate":
            content = ["Non-Creamy Layer Certificate", "This certifies that the person does not belong to Creamy Layer."]
        elif oos_type == "Income_Certificate":
            content = ["Income Certificate", "Annual Family Income is Rs. 4,00,000/-"]
        else:
            content = ["Domicile Certificate", "Resident of Maharashtra for 15 years."]
            
        y += 80
        for line in content:
            draw.text((50, y), line, fill="black", font=text_font)
            y += 40

    # Apply degradation based on index to some images
    if index % 2 == 1:
        img = apply_degradations(img)
        
    return img

def main():
    print("Setting up directories...")
    setup_directories()
    
    print("Generating demo images...")
    generated_files = {cls: [] for cls in CLASSES}
    
    for cls in CLASSES:
        num_images = 9 if cls != "UNKNOWN_OUT_OF_SCOPE" else 3
        for i in range(num_images):
            img = create_document(cls, i, i)
            filename = f"{cls}_demo_{i}.jpg"
            filepath = DEMO_DIR / cls / filename
            img.save(filepath, "JPEG", quality=85)
            generated_files[cls].append(filepath)
            
    print("Copying demo images to original folders for candidates...")
    
    for cand, cand_classes in CANDIDATES.items():
        for i, cls in enumerate(cand_classes):
            if generated_files[cls]:
                src = generated_files[cls][i % len(generated_files[cls])]
                dst = ORIGINALS_DIR / cand / src.name
                shutil.copy2(src, dst)
                
    print("Demo data generation complete!")

if __name__ == "__main__":
    main()
