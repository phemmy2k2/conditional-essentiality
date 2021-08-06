from django.urls import path
from . import views
# from django.conf import settings
# from django.conf.urls.static import static

app_name = "contact"
urlpatterns = [
    path("", views.contact, name="contact"),
    path("predict", views.contact, name="predict"),
    path("loader", views.loader, name="loader"),
]

# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)