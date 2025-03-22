from django.db import models
from django.contrib.auth.models import User

class ChatMessage(models.Model):
    ROLE_CHOICES = (
        ('user', 'User'),
        ('bot', 'Bot'),
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name="chat_messages",
        help_text="The user who sent or received the message."
    )
    message = models.TextField(help_text="The content of the chat message.")
    timestamp = models.DateTimeField(auto_now_add=True, help_text="Time the message was created.")

    def __str__(self):
        return f"{self.user.username} [{self.timestamp:%Y-%m-%d %H:%M:%S}]: {self.message[:50]}"