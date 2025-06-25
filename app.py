import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from dotenv import load_dotenv
import uuid
from functools import wraps
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import StringIO

# Initialize app and environment
app = Flask(__name__)
load_dotenv()

# Constants
DEFAULT_MODEL = "llama3-70b-8192"
MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Contact Information
DEFAULT_CONTACT = {
    "email": "nutrition@healthplanner.com",
    "phone": "+1 (555) 123-4567",
    "website": "https://healthplanner.com",
    "booking": "https://healthplanner.com/consultation"
}

# Error Classes
class DietPlanError(Exception):
    pass

class InvalidInputError(DietPlanError):
    pass

class GroqServiceError(DietPlanError):
    pass

# Helpers
def validate_input(data: Dict[str, Any]) -> None:
    required_fields = {
        'name': str,
        'age': int,
        'weight': (int, float),
        'height': (int, float),
        'activityLevel': str,
        'healthGoal': str
    }

    missing = [field for field in required_fields if field not in data]
    if missing:
        raise InvalidInputError(f"Missing required fields: {', '.join(missing)}")

    try:
        data['age'] = int(data['age'])
        data['weight'] = float(data['weight'])
        data['height'] = float(data['height'])
    except (ValueError, TypeError):
        raise InvalidInputError("Age, weight, and height must be numeric")

    if not 1 <= data['age'] <= 120:
        raise InvalidInputError("Age must be between 1 and 120")
    if data['weight'] <= 0 or data['height'] <= 0:
        raise InvalidInputError("Weight and height must be positive values")

def build_prompt(data: Dict[str, Any]) -> str:
    return f"""
    **Detailed 7-Day Diet Plan Request**
    
    **User Profile:**
    - Name: {data['name']}
    - Age: {data['age']} | Weight: {data['weight']}kg | Height: {data['height']}cm
    - Primary Goal: {data['healthGoal']}
    - Activity Level: {data['activityLevel']}
    - Dietary Preferences: {data.get('dietaryPreferences', 'None')}
    - Preferred Cuisine: {data.get('preferredCuisine', 'Flexible')}
    - Allergies: {data.get('allergies', 'None')}
    - Health Conditions: {data.get('existingHealthConditions', 'None')}

    **Requirements:**
    1. Create a complete 7-day meal plan
    2. Include breakfast, lunch, dinner + 2 snacks daily
    3. Specify portion sizes and preparation instructions
    4. Provide macronutrient breakdown for each meal
    5. Include weekly grocery shopping list
    6. Add nutrition tips for achieving {data['healthGoal']}
    7. Format in clear markdown with daily sections
    """

def send_email(recipient_email: str, subject: str, body: str) -> None:
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASSWORD")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        logger.info(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")

def json_response(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            start_time = datetime.now(timezone.utc)
            result = func(*args, **kwargs)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            if isinstance(result, tuple):
                data, status = result
            else:
                data, status = result, 200

            response = {
                "status": "success",
                "data": data,
                "meta": {
                    "request_id": str(uuid.uuid4()),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "execution_time": f"{execution_time:.2f}s"
                }
            }
            return jsonify(response), status

        except InvalidInputError as e:
            logger.warning(f"Input validation failed: {str(e)}")
            return jsonify({
                "status": "error",
                "message": str(e),
                "code": "invalid_input"
            }), 400

        except GroqServiceError as e:
            logger.error(f"Groq service error: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Diet generation service unavailable",
                "code": "service_unavailable"
            }), 503

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "message": "Internal server error",
                "code": "server_error"
            }), 500

    return wrapper

# Initialize Groq client
try:
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=groq_api_key)
    logger.info("Groq client initialized successfully")

except Exception as e:
    logger.critical(f"Failed to initialize Groq client: {str(e)}")
    raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-diet-plan', methods=['POST'])
@json_response
def generate_diet_plan():
    if not request.is_json:
        raise InvalidInputError("Request must be JSON")

    data = request.get_json()

    validate_input(data)
    
    prompt = build_prompt(data)
    logger.info(f"Generated prompt for {data['name']}")

    try:
        response = client.chat.completions.create(
            model=os.getenv('GROQ_MODEL', DEFAULT_MODEL),
            messages=[{"role": "system", "content": "You are an expert nutritionist."},
                      {"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        diet_plan = response.choices[0].message.content

        # Send email if email provided
        recipient = data.get("email")
        if recipient:
            try:
                subject = f"Your 7-Day Diet Plan, {data['name']}"
                send_email(recipient, subject, diet_plan)
            except Exception as email_err:
                logger.warning(f"Email could not be sent to {recipient}: {email_err}")

        return {
            "plan": diet_plan,
            "contact": {
                "email": data.get("nutritionistEmail", DEFAULT_CONTACT["email"]),
                "phone": data.get("nutritionistContact", DEFAULT_CONTACT["phone"]),
                "website": DEFAULT_CONTACT["website"],
                "booking": data.get("consultationLink", DEFAULT_CONTACT["booking"])
            }
        }

    except Exception as e:
        logger.error(f"Groq API call failed: {str(e)}")
        raise GroqServiceError("Failed to generate diet plan")

@app.route('/download-diet-plan', methods=['POST'])
def download_diet_plan():
    if not request.is_json:
        return jsonify({"error": "Invalid input"}), 400

    data = request.get_json()
    try:
        validate_input(data)
        prompt = build_prompt(data)

        response = client.chat.completions.create(
            model=os.getenv('GROQ_MODEL', DEFAULT_MODEL),
            messages=[{"role": "system", "content": "You are an expert nutritionist."},
                      {"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        diet_plan = response.choices[0].message.content

        # Create file-like object
        file_content = StringIO(diet_plan)
        file_content.seek(0)

        return send_file(file_content, as_attachment=True, download_name="diet_plan.txt", mimetype="text/plain")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
