from django.shortcuts import render

from django.http import HttpResponse

from django.template import loader
from .models import collection

def index(request):
    return HttpResponse("Hello, world. You're at Mythos")

def home(request):
   text = collection.objects.all()
   context = {'collection':text}

   return render(request,'publicmood/home2.html',{})