from io import BytesIO
from flask import Response, send_file
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

def generate_csv(df, device_types, history_range):
    df['timestamp']=df['timestamp'].astype(str)
    csv_data = df.to_csv(index=False)
    devices_str = "-".join(device_types)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=energy_report_{devices_str}_for_{history_range}.csv"
        }
    )

def generate_pdf(df, device_types, history_range):
    devices_str = "-".join(device_types)
    filename = f"energy_report_{devices_str}_for_{history_range}.pdf"

    buffer = BytesIO()

    pdf = SimpleDocTemplate(buffer)

    data = [df.columns.tolist()] + df.values.tolist()

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))

    pdf.build([table])
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )