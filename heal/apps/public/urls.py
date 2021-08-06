from django.urls import path
from . import views
# from django.conf import settings
# from django.conf.urls.static import static

app_name = "public"
urlpatterns = [
    path("", views.index, name="index"),
    path("about", views.about, name="about"),
    path("dataset", views.dataset, name="dataset"),
]

# if settings.DEBUG:
#     # urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_URL)
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)