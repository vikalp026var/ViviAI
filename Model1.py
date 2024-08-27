import os
import cv2
from flask import request, render_template, url_for, Flask, redirect, Blueprint, send_file
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import seaborn as sns
from io import BytesIO
from Vivi_AI.utils.main_utils import fractal_dimension, lacunarity, entropy
from Vivi_AI.logger import logging
from Vivi_AI.exception import CustomException
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

model1 = Blueprint(__name__, "model1")
import matplotlib
matplotlib.use('Agg')
# Set the upload folder
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

df1 = pd.read_csv(os.path.join("data_transformation", "feature.csv"))
df1['standard_label'] = df1['filename'].apply(lambda x: 'Normal' if 'Normal' in x else 'GBM-Grade IV')


def clean_upload_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f'Error: {e}')


def create_feature_plot(df1, fd, en, lc):
    features = ['fractal_dimension', 'lacunarity', 'entropy']
    test_values = [fd, lc, en]
    titles = ['Fractal Dimension', 'Lacunarity', 'Entropy']
    plot_paths = []

    for feature, test_value, title in zip(features, test_values, titles):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=df1, x=feature, hue='standard_label', element='step', stat='density', common_norm=False,
                     palette={'Normal': 'blue', 'GBM-Grade IV': 'red'}, ax=ax)
        ax.axvline(x=test_value, color='green', linestyle='--', label='Test Image')
        ax.set_title(f'Distribution of {title} by Category (Normal and GBM)')
        ax.set_xlabel(title)
        ax.set_ylabel('Density')

        # Custom legend
        handles, labels = ax.get_legend_handles_labels()
        custom_handles = [
            plt.Line2D([0], [0], color='blue', lw=4, label='Normal'),
            plt.Line2D([0], [0], color='red', lw=4, label='GBM-Grade IV'),
            plt.Line2D([0], [0], color='green', linestyle='--', lw=2, label='Test Image')
        ]
        ax.legend(handles=custom_handles, title='Category', loc='upper right')

        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{feature}_plot.png')
        fig.savefig(plot_path)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths



def predict1(img_file):
    try:
        # Clean the upload folder before saving new files
        clean_upload_folder(app.config['UPLOAD_FOLDER'])

        # Load the model
        model_path = os.path.join("trained_model","model.pkl")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Get the file from the form
        if not img_file:
            return render_template('research.html', result="No file uploaded",
                                    fractal_dimension=None, entropy=None, lacunarity=None)

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(file_path)

        # Read the image
        img = Image.open(file_path)
        img = np.array(img.convert('L'))  # Convert image to grayscale

        # Compute features
        fd = fractal_dimension(img)
        en = entropy(img)
        lc = lacunarity(img, window_size=8)

        # Ensure features are single numerical values
        fd = fd[0] if isinstance(fd, tuple) else fd
        en = en[0] if isinstance(en, tuple) else en
        lc = lc[0] if isinstance(lc, tuple) else lc

        # Create DataFrame and ensure correct types
        test = pd.DataFrame(data={
            'fractal_dimension': [fd],
            'entropy': [en],
            'lacunarity': [lc]
        }).astype(float)

        # Predict
        prediction = model.predict(test.values)

        # Interpret the result
        result = "GBM" if prediction[0] == 1 else "Normal"

        # Create the feature plot
        plot_paths = create_feature_plot(df1, fd, en, lc)
        return result,plot_paths,fd,en,lc

    except Exception as e:
        logging.error(e)



@model1.route('/Model1/download_report', methods=['GET'])
def download_report():
    result = request.args.get('result')
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Check if the style exists before adding it
    if 'Centered' not in styles:
        styles.add(ParagraphStyle(name='Centered', alignment=1, fontSize=20, spaceAfter=20))
    if 'Heading' not in styles:
        styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=10, textColor=colors.black))
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=12, spaceAfter=10))
    if 'Result' not in styles:
        styles.add(ParagraphStyle(name='Result', fontSize=18, spaceAfter=20, textColor=colors.red, fontName="Helvetica-Bold"))

    # Add logos
    vivi_logo_path = 'static/vivi.png'
    elements.append(RLImage(vivi_logo_path, width=200, height=100))
    elements.append(Spacer(5, 12))

    # Add title
    elements.append(Paragraph("GBM Cancer Detection Report", styles['Centered']))

    # Add Introduction
    elements.append(Paragraph("Introduction", styles['Heading']))
    elements.append(Paragraph("This report provides an analysis of GBM cancer detection and patient outcomes.", styles['BodyText']))

    # Add Conclusion
    elements.append(Paragraph("Conclusion:", styles['Heading']))
    elements.append(Paragraph(f"The analysis indicates the result as: {result}", styles['Result']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='GBM_Cancer_Detection_Report.pdf', mimetype='application/pdf')


app.register_blueprint(model1, url_prefix='/model1')
