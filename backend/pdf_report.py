import os
from datetime import datetime
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    PageBreak,
)


def generate_pdf_report(
    prediction_id,
    image_name,
    original_image_path,
    predicted_disease,
    confidence,
    gradcam_image_path,
    model_version="EfficientNetB0",
    patient_name="N/A",
    patient_age="N/A",
    patient_place="N/A",
    patient_id=None,
    examination_id="N/A",
    doctor_name="N/A",
    staff_username="N/A",
    examined_at=None,
    **kwargs,
):
    """Generate a standardized clinical PDF report for VisionX AI screening results."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(project_root, "backend", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    report_id = f"VX-{prediction_id}-{uuid4().hex[:8].upper()}"
    timestamp = datetime.now().astimezone()

    pdf_filename = f"{report_id}.pdf"
    pdf_path = os.path.join(reports_dir, pdf_filename)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "VisionXTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#087E8B"),
        fontSize=22,
        spaceAfter=6,
    )

    heading_style = ParagraphStyle(
        "VisionXHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#087E8B"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = styles["BodyText"]

    story = []

    # Report header
    story.append(Paragraph("VisionX AI", title_style))
    story.append(
        Paragraph(
            "AI-Assisted Multi-Disease Retinal Screening Report",
            ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                fontSize=11,
            ),
        )
    )
    story.append(Spacer(1, 14))

    # Patient & Examination Metadata Table
    exam_date_str = examined_at.strftime("%d %b %Y, %I:%M %p") if isinstance(examined_at, datetime) else str(examined_at or timestamp.strftime("%d %b %Y"))

    metadata = [
        ["Report Reference", report_id, "Screening Timestamp", timestamp.strftime("%d %b %Y, %I:%M %p")],
        ["Patient ID", f"#{patient_id}" if patient_id is not None else "N/A", "Patient Name", str(patient_name)],
        ["Patient Age", f"{patient_age} Years" if patient_age != "N/A" else "N/A", "Patient Place", str(patient_place)],
        ["Examination ID", f"#{examination_id}" if examination_id != "N/A" else "Walk-in Intake", "Assigned Doctor", f"Dr. {doctor_name}" if doctor_name != "N/A" else "Clinical Pool"],
        ["Processing Staff", str(staff_username), "Uploaded File", str(image_name)],
    ]

    metadata_table = Table(metadata, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF5F7")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EAF5F7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 14))

    # Prediction summary
    story.append(Paragraph("1. AI Diagnostic Screening Result", heading_style))

    result_data = [
        ["Detected Condition / Category", str(predicted_disease)],
        ["AI Model Confidence Score", f"{float(confidence):.2f}%"],
        ["Neural Network Architecture", str(model_version)],
    ]

    result_table = Table(result_data, colWidths=[2.5 * inch, 4.7 * inch])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF5F7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]
        )
    )

    story.append(result_table)
    story.append(Spacer(1, 12))

    # Original Fundus Scan
    story.append(Paragraph("2. Uploaded Retinal Fundus Scan", heading_style))

    if os.path.isfile(original_image_path):
        original = PDFImage(original_image_path, width=3.2 * inch, height=3.2 * inch)
        original.hAlign = "LEFT"
        original._restrictSize(3.2 * inch, 3.2 * inch)
        story.append(original)
    else:
        story.append(Paragraph("Original fundus scan file was not found on disk.", body_style))

    story.append(Spacer(1, 10))

    # Page two
    story.append(PageBreak())

    story.append(Paragraph("3. Grad-CAM Activation Heatmap (Visual Evidence)", heading_style))

    if os.path.isfile(gradcam_image_path):
        heatmap = PDFImage(gradcam_image_path, width=4.5 * inch, height=4.5 * inch)
        heatmap.hAlign = "LEFT"
        heatmap._restrictSize(4.5 * inch, 4.5 * inch)
        story.append(heatmap)
    else:
        story.append(Paragraph("Grad-CAM attention heatmap file was not found on disk.", body_style))

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights convolutional "
            "feature activations that influenced the model's prediction. Warm regions (red/yellow) "
            "indicate salient ocular landmarks corresponding to the detected condition.",
            body_style,
        )
    )

    story.append(Paragraph("4. Recommended Clinical Protocol", heading_style))
    story.append(
        Paragraph(
            "This report constitutes preliminary screening decision-support. A qualified "
            "ophthalmologist must review this examination alongside direct ophthalmoscopy, "
            "OCT scanning, or slit-lamp biomicroscopy before establishing a definitive clinical diagnosis.",
            body_style,
        )
    )

    story.append(Spacer(1, 10))
    story.append(Paragraph("Clinical Safety Notice", heading_style))
    story.append(
        Paragraph(
            "AI results are intended to support screening and clinical review. "
            "Changes in predictions or confidence scores do not independently establish "
            "clinical improvement or deterioration. A qualified eye specialist should "
            "interpret the findings and determine appropriate next steps. "
            "VisionX AI is an educational and research-driven retinal screening prototype; "
            "its output does not constitute a confirmed medical diagnosis.",
            body_style,
        )
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="VisionX AI Screening Report",
        author="VisionX AI",
    )

    doc.build(story)

    return pdf_path