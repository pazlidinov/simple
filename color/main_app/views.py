from typing import Any
from django.shortcuts import render, redirect
from django.views.generic import View, TemplateView, DetailView
from django.conf import settings
from .forms import PhotoForm
from .util_2 import main
from .models import Photo

# Create your views here.


# View for Home page
class HomePageView(TemplateView):
    template_name = "index.html"


# Show to modified images
class ImgPageView(DetailView):
    model = Photo
    template_name = "modified.html"


# Modified images
class ModifiedImgView(View):

    def post(self, request):
        form = PhotoForm(request.POST, request.FILES)

        if form.is_valid():
            f = form.save(commit=False)
            f.title = f.image.name
            f.save()
            pk = f.id

            main(
                f".{f.image.url}",
                f"./media/modified_img_{pk}.png",
                int(request.POST.get("num_clusters"))
            )

            photo = Photo.objects.get(id=pk)
            photo.modified_image = f"modified_img_{pk}_with_colors.png"
            photo.all_files = f"modified_img_{pk}.zip"
            photo.save()
            return redirect("main_app:img", pk=pk)
        return redirect("main_app:home")
