from django.shortcuts import render, HttpResponseRedirect, redirect
# from django.urls import reverse
from .forms import ContactForm
from random import randint

def contact(request):
    randn = randint(1,10000000)
    print(randn)
    if request.method == 'GET':
        form = ContactForm()
        form.fields['userId'].initial = randn
        # form.fields['result_path'].initial = 'pred_' + str(randn) + '.csv'
        thresholds = [0.70,0.75,0.80,0.85,0.90,0.95]
        return render(request, "contact.html", {'form': form,'thresholds':thresholds, 'disp':'d-none'})

    elif request.method == 'POST'and "load_file" in request.POST :
        if request.POST.get("load_file"):
            form = ContactForm(request.POST, request.FILES)
            if form.is_valid():
                # clean data from the Filefield before save
                if request.FILES:
                    form.save()

                    # import function to run
                    from .scripts.annotate import main
                    # call function
                    labfile = request.FILES['labfile_path']
                    unlabfile = request.FILES['unlabfile_path']
                    thres = request.POST['threshold']

                    res = main(randn, labfile, unlabfile, thres)
                    # return user to required page
                    download_file = res['download_file']
                    _ = res.pop('download_file')

                    context = {'form': form, 'result':res, 'download_file':download_file, 'disp':''}  #
                    return render(request, 'predict.html', context)

                else: # ensures files are attached before submitting to server
                    form = ContactForm()
                    context = {'form': form, 'result': {'Error':'Select files to process'}, 'disp':'d-none'}
                    return render(request, 'contact.html', context)
            else: # ensures contents in form fields are valid
                if request.POST['labdata'] == '' and request.POST['unlabdata'] == '':
                    form = ContactForm()
                    context = {'form': form, 'result': {'Error': 'Enter appropriate input...'}, 'disp':'d-none'}
                    return render(request, 'contact.html', context)
                else:
                    print(request.POST['labdata'])
                    print(request.POST['unlabdata'])
                    # import function to run
                    from .scripts.annotate_post_description import main
                    # call function
                    labfile = request.POST['labdata']
                    unlabfile = request.POST['unlabdata']
                    thres = request.POST['threshold']

                    res = main(randn, labfile, unlabfile, thres)
                    # return user to required page
                    download_file = res['download_file']
                    _ = res.pop('download_file')

                    context = {'form': form, 'result': res, 'download_file': download_file, 'disp': ''}  #
                    return render(request, 'predict.html', context)
    elif request.method == 'POST' and "desc_stats" in request.POST:
        if request.POST.get("desc_stats"):
            form = ContactForm(request.POST, request.FILES)
            if form.is_valid():
                # clean data from the Filefield before save
                if request.FILES:
                    form.save()

                    # import function to run
                    from .scripts.descriptive import main
                    # call function
                    labfile = request.FILES['labfile_path']
                    unlabfile = request.FILES['unlabfile_path']
                    # print(labfile)
                    res = main(labfile, unlabfile)
                    # set the path of the input files after loading
                    res['lab'] = labfile
                    res['unlab'] = unlabfile

                    context = {'form': form, 'result': res, 'disp': ''}  #
                    return render(request, 'contact.html', context)
                else: # ensures files are attached before submitting to server
                    form = ContactForm()
                    context = {'form': form, 'result': {'Error':'Select files to process'}}
                    return render(request, "contact.html", context)
            else:  # ensures contents in form fields are valid
                form = ContactForm()
                context = {'form': form, 'result': {'Error': 'Enter appropriate input...'}}
                return render(request, 'contact.html', context)
    else:
        # form = ContactForm()
        # context = {'form': form, 'result': {'Error': 'Select files to process'}, 'disp': 'd-none'}
        # return render(request, 'contact.html', context)
        print(request.POST)
        return NotImplementedError


def loader(request):
    return render(request, "loader.html")
