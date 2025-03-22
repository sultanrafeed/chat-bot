from django.contrib import admin
from .models import ChatMessage

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("user", "message", "timestamp")
    list_filter = ("user",)
    search_fields = ("message",)
    date_hierarchy = "timestamp"