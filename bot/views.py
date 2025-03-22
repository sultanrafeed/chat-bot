import json
from django.shortcuts import render, redirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

# Import our chatbot agent and chat history handler
from .agent import ChatbotAgent
from .chatstore import ChatHistoryHandler
from .models import ChatMessage

# Initialize global instances
chatbot_agent = ChatbotAgent()
chat_history_handler = ChatHistoryHandler()

def home(request):
    if request.user.is_authenticated:
        return redirect("chat")
    return redirect("login")

def chat(request):
    if not request.user.is_authenticated:
        return redirect("login")
    # Use the logged-in user's username as the session identifier.
    session_id = request.user.username
    history = chat_history_handler.get_chat_history(session_id)
    context = {"history": history, "user": request.user}
    return render(request, "chat.html", context)

@csrf_exempt
def get_response(request):
    if request.method == "POST":
        if not request.user.is_authenticated:
            return HttpResponse(json.dumps({"error": "Unauthorized"}), content_type="application/json", status=401)
        try:
            data = json.loads(request.body)
            user_message = data.get("userMessage", "")
        except Exception:
            return HttpResponse(json.dumps({"error": "Invalid input"}), content_type="application/json", status=400)
        session_id = request.user.username
        # Save the user's message in storage (e.g., Firestore or ORM)
        chat_history_handler.add_query(session_id, user_message)
        # Get the bot response
        bot_response = chatbot_agent.get_response(session_id, user_message)
        # Save the bot response
        chat_history_handler.add_response(session_id, bot_response)
        return HttpResponse(json.dumps({"response": bot_response}), content_type="application/json")
    return HttpResponse("Method not allowed", status=405)

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("username").lower().strip()
        password = request.POST.get("password")
        user = authenticate(request, username=email, password=password)
        if user:
            auth_login(request, user)
            return redirect("chat")
        messages.error(request, "Invalid email or password.")
    return render(request, "login.html")


def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            email = request.POST.get("email")
            user.username = email.lower().strip()
            user.email = email.lower().strip()
            user.save()
            messages.success(request, "Account created. Please log in.")
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "register.html", {"form": form})

def signout(request):
    auth_logout(request)
    return redirect("login")