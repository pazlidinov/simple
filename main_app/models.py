from django.db import models


# Create your models here.
class Photo(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to="orginal_photos/")
    modified_image = models.ImageField(upload_to="modified_photos/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    all_files = models.FileField(upload_to='zips/')

    def __str__(self):
        return self.title
