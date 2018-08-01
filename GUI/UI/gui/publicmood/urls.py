from django.contrib import admin
from django.urls import include, path
from . import views
from django.conf.urls import url

urlpatterns = [
    url(r'^', views.home, name = 'home'),
    #url(r'^admin/$', admin.site.urls)
]
