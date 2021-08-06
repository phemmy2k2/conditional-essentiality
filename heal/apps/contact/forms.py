from django import forms
from .models import infile

class ContactForm(forms.ModelForm):
    userId = forms.CharField(required=True, label="", widget=forms.TextInput(attrs={'class':'d-none'})) #
    labfile_path = forms.FileField(required=False, label="1. Select labeled data") # , widget=forms.FileField()
    unlabfile_path = forms.FileField(required=False, label="2. Select Unlabeled data") # , widget=forms.FileField()
    # result_path = forms.CharField(required=False, label="")
    # txtlabfile_path = forms.CharField(required=False, label="labfile")
    # txtunlabfile_path = forms.CharField(required=False, label="", widget=forms.TextInput(attrs={'class': 'd-none'}))

    class Meta:
        model = infile
        fields = ('userId', 'labfile_path','unlabfile_path') #,'result_path'

