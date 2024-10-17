from django.urls import path

from . import views

app_name = "main_app"

urlpatterns = [
    path("", views.HomePageView.as_view(), name="home"),
    path("modified/", views.ModifiedImgView.as_view(), name="modified"),
    path("img/<pk>", views.ImgPageView.as_view(), name="img"),
]
