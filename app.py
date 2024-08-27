import os
import logging
from flask import Flask, render_template, request, redirect, url_for, session
from Vivi_AI.utils.main_utils import final_feature_Extr, list_
from auth import auth_bp,oauth
from chat_app import chatbot_bp
from Model1 import model1, predict1
from Model2 import model2, predict2

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
oauth.init_app(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Register the authentication blueprint
app.register_blueprint(auth_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(model1)
app.register_blueprint(model2)

@app.route("/")
def index():
    app.logger.info('Index route called')
    if 'google_token' in session or 'user' in session:
        app.logger.info('User authenticated, rendering index.html')
        return render_template("index.html")
    app.logger.info('No authenticated user found, redirecting to login')
    return redirect(url_for('auth.login'))

@app.route('/research', methods=['GET', 'POST'])
def research():
    if 'google_token' not in session and 'user' not in session:
        app.logger.info('User not authenticated, redirecting to login')
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        try:
            img_file = request.files.get('file')
            result,plot_paths,fd,en,lc=predict1(img_file)
            session['plot_paths'] = plot_paths
            session['fd'] = fd
            session['en'] = en
            session['lc'] = lc
               
            return render_template('research.html', result=result)
           
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            if 'model' in str(e).lower():
                return render_template('research.html', result=f"An error occurred while loading the model: {str(e)}")
            elif 'file' in str(e).lower():
                return render_template('research.html', result=f"An error occurred while processing the file: {str(e)}")
            else:
                return render_template('research.html', result=f"An unexpected error occurred: {str(e)}")
    else:  # GET request
        return render_template('research.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/graphs')
def graphs():
    plot_paths = session.get('plot_paths')
    fd = session.get('fd')
    en = session.get('en')
    lc = session.get('lc')

    # Check if data is available
    if not plot_paths or not fd or not en or not lc:
        # Handle the case where the data is missing
        return render_template('graph.html', error="No graph data available.")

    # Render graph.html with the retrieved data
    return render_template('graph.html', plot_paths=plot_paths, fd=fd, en=en, lc=lc)

@app.route('/contact')
def contact():
    if 'google_token' in session or 'user' in session:
        app.logger.info('User authenticated, rendering contact.html')
        return render_template("contact.html")
    app.logger.info('No authenticated user found, redirecting to login')
    return redirect(url_for('auth.login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
