from flask import Flask, render_template, request, send_file
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage


import pandas as pd
import os
import cv2

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# -------------------- LOAD DATASET --------------------
df = pd.read_csv("image_features.csv")


# -------------------- CONFIDENCE & SEVERITY --------------------
def get_confidence_severity(T, I):
    if T >= 80 and I <= 10:
        return "High", "Severe"
    elif T >= 60 and I <= 20:
        return "Medium", "Moderate"
    else:
        return "Low", "Minor"


# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(image_name):
    input_path = os.path.join("static", "images", image_name)
    output_dir = os.path.join("static", "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, image_name)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    processed = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    cv2.imwrite(output_path, processed)
    return output_path





# -------------------- HOME --------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------- DASHBOARD --------------------
@app.route("/dashboard")
def dashboard():
    total = len(df)
    defect = len(df[df["Label"] == "defect"])
    normal = len(df[df["Label"] == "normal"])

    avg_T = round(df["Truth"].mean(), 2)
    avg_I = round(df["Indeterminacy"].mean(), 2)
    avg_F = round(df["Falsity"].mean(), 2)

    return render_template(
        "dashboard.html",
        total=total,
        defect=defect,
        normal=normal,
        avg_T=avg_T,
        avg_I=avg_I,
        avg_F=avg_F
    )


# -------------------- IMAGE-WISE ANALYSIS --------------------
@app.route("/image_analysis", methods=["GET", "POST"])
def image_analysis():
    images = df["Image_Name"].tolist()
    selected = images[0]

    if request.method == "POST":
        selected = request.form["image"]

    row = df[df["Image_Name"] == selected].iloc[0]

    confidence, severity = get_confidence_severity(
        row["Truth"], row["Indeterminacy"]
    )

    preprocess_image(selected)

    # ✅ ADD HERE
    explanation_text = (
        f"The Truth value of {row['Truth']}% indicates the degree of defect "
        f"presence on the surface. Indeterminacy of {row['Indeterminacy']}% "
        f"represents uncertainty caused by surface texture variation, noise, "
        f"or illumination effects. The Falsity value of {row['Falsity']}% "
        f"corresponds to normal surface characteristics. "
        f"Based on neutrosophic evaluation, the surface is classified as "
        f"{row['Label'].upper()} with {confidence} confidence and "
        f"{severity} severity."
    )

    result = {
        "Truth": row["Truth"],
        "Indeterminacy": row["Indeterminacy"],
        "Falsity": row["Falsity"],
        "Label": row["Label"],
        "Confidence": confidence,
        "Severity": severity
    }

    return render_template(
        "image_analysis.html",
        images=images,
        selected=selected,
        result=result,
        image_path=f"/static/images/{selected}",
        processed_path=f"/static/processed/{selected}",
        explanation=explanation_text   # ✅ PASS HERE
    )


# -------------------- PDF REPORT --------------------
@app.route("/download_report/<image_name>")
def download_report(image_name):
    row = df[df["Image_Name"] == image_name].iloc[0]

    confidence, severity = get_confidence_severity(
        row["Truth"], row["Indeterminacy"]
    )

    file_path = f"Neutrosophic_Report_{image_name}.pdf"
    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    # -------------------- TITLE --------------------
    content.append(Paragraph(
        "<font size=18><b>Neutrosophic Surface Defect Analysis Report</b></font>",
        styles["Title"]
    ))
    content.append(Spacer(1, 20))

    # -------------------- BASIC INFO --------------------
    info_table = Table([
        ["Image Name", image_name],
        ["Classification", row["Label"].upper()],
        ["Confidence Level", confidence],
        ["Severity Level", severity]
    ], colWidths=[150, 300])

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("FONT", (0,0), (-1,-1), "Helvetica"),
        ("FONT", (0,0), (0,-1), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))

    content.append(info_table)
    content.append(Spacer(1, 25))

    # -------------------- NEUTROSOPHIC VALUES --------------------
    content.append(Paragraph(
        "<b>Neutrosophic Membership Values</b>",
        styles["Heading2"]
    ))
    content.append(Spacer(1, 10))

    tif_table = Table([
        ["Measure", "Value (%)"],
        ["Truth (T)", f"{row['Truth']}%"],
        ["Indeterminacy (I)", f"{row['Indeterminacy']}%"],
        ["Falsity (F)", f"{row['Falsity']}%"]
    ], colWidths=[200, 250])

    tif_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ]))

    content.append(tif_table)
    content.append(Spacer(1, 25))
    # -------------------- IMAGE VISUALIZATION --------------------
    content.append(Paragraph(
        "<b>Surface Image Visualization</b>",
        styles["Heading2"]
    ))
    content.append(Spacer(1, 10))

    original_img_path = os.path.join("static", "images", image_name)
    processed_img_path = os.path.join("static", "processed", image_name)

    img_table_data = []

    # Original image
    if os.path.exists(original_img_path):
        original_img = RLImage(original_img_path, width=180, height=180)
        img_table_data.append(["Original Image", original_img])
    else:
        img_table_data.append(["Original Image", "Not Available"])

    # Preprocessed image
    if os.path.exists(processed_img_path):
        processed_img = RLImage(processed_img_path, width=180, height=180)
        img_table_data.append(["Preprocessed Image", processed_img])
    else:
        img_table_data.append(["Preprocessed Image", "Not Available"])

    img_table = Table(img_table_data, colWidths=[160, 300])
    img_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("FONT", (0,0), (0,-1), "Helvetica-Bold"),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))

    content.append(img_table)
    content.append(Spacer(1, 25))

       
    # -------------------- EXPLANATION --------------------
    content.append(Paragraph(
        "<b>Interpretation & Explanation</b>",
        styles["Heading2"]
    ))
    content.append(Spacer(1, 10))

    explanation = f"""
    The analyzed surface image exhibits a Truth membership value of
    {row['Truth']}%, indicating the presence of defect characteristics.
    The Indeterminacy value of {row['Indeterminacy']}% reflects uncertainty
    due to surface texture variation or illumination noise. The Falsity
    value of {row['Falsity']}% represents normal surface behavior.

    Based on neutrosophic evaluation, the surface is classified as
    <b>{row['Label'].upper()}</b> with <b>{confidence}</b> confidence and
    <b>{severity}</b> severity.
    """

    content.append(Paragraph(explanation, styles["Normal"]))
    content.append(Spacer(1, 30))

    # -------------------- FOOTER --------------------
    content.append(Paragraph(
        "<font size=9>Department of Information Technology<br/>"
        "Academic Year: 2024–2025</font>",
        styles["Normal"]
    ))

    doc.build(content)
    return send_file(file_path, as_attachment=True)

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
