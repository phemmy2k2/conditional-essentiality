from django.db import models

# Create your models here.
class infile(models.Model):
    userId = models.CharField(max_length=200, unique=True)
    labfile_path = models.FileField(max_length=200, upload_to='indir', default='default.csv') #, storage='', default=''
    unlabfile_path = models.FileField(max_length=200, upload_to='indir', default='default.csv') #, storage='', default=''
    result_path = models.CharField(max_length=200, default='default.csv')