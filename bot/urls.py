from django.urls import path
from .views import home, login_view, register, signout, chat, get_response

urlpatterns = [
    path("", home, name="home"),
    path("login/", login_view, name="login"),
    path("register/", register, name="register"),
    path("signout/", signout, name="signout"),
    path("chat/", chat, name="chat"),
    path("api/get-response/", get_response, name="get_response"),
]