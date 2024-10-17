# forms.py
from django import forms
from .models import Photo


# Form for Photo model
class PhotoForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ["image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["image"].widget.attrs.update({"class": "form-control-file"})
