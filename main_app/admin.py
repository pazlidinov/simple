from django.contrib import admin
from django.utils.html import mark_safe
from .models import Photo


# Register your models here.
@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ["title", "org_img"]
    readonly_fields = ["org_img"]

    def org_img(self, obj):
        return mark_safe(f'<img src="{obj.image.url}" width="100" height="100" />')

    # def mod_img(self, obj):
    #     return mark_safe(f'<img src="{obj.modified_image.url}" width="100" height="100" />')
