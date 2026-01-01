from django.shortcuts import render , redirect,get_object_or_404
from django.http import HttpResponse

from .models import food , Review , user_details
from itemCart.models import cartSes , order1 , orderedItems,Contact
from .forms import foodForm , user_detailsForm

from . import machine

from django.db.models import Q
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate , login , logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm

# from django.http import HttpResponse
# from django.contrib.auth.models import User

# Create your views here.

def home(req):
    contact=Contact.objects.first()
    choice = food.course_choice
    rate = Review.objects.all()
    q = req.GET.get('q') if req.GET.get('q') != None else ''
    item = food.objects.filter(Q(course__icontains = q))
    
    l = machine.seggestions(req)
    
    context = {"food" : item , "course" : choice , "rate" : rate , 'l' : l,"contact":contact}
    return render(req,'home.html',context)

def loginPage(req):
    page = 'login'

    if req.user.is_authenticated:
        return redirect('home')

    if req.method == "POST":
        username = req.POST.get('username').lower()
        password = req.POST.get('password')

        user = authenticate(req, username=username, password=password)

        if user is not None:
            login(req, user)
            messages.success(req, "Login successful")
            if user.is_superuser:
                return redirect('dashboard')
            return redirect('home')
        else:
            messages.error(req, "Invalid username or password")

    return render(req, 'login-register.html', {'page': page})


def logoutPage(req):
    logout(req)
    return redirect('home')

def registerPage(req):
    page = 'register'
    form = UserCreationForm()

    if req.method == "POST":
        form = UserCreationForm(req.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()

            # ✅ CREATE user_details WITH REQUIRED FIELDS
            user_details.objects.create(
                user=user,
                email=req.POST.get('email'),
                phone=req.POST.get('phone'),
                address=req.POST.get('address'),
            )

            cartSes.objects.create(user=user)

            login(req, user)
            messages.success(req, "Registration successful")
            return redirect('user_details_form')

        else:
            messages.error(req, "Registration failed. Please check the form.")

    return render(req, 'login-register.html', {
        'form': form,
        'page': page
    })



def user_deltail_Form(req):
    user_for_det, created = user_details.objects.get_or_create(
        user=req.user,
        defaults={
            'email': req.user.email or '',
            'phone': 0,
            'address': ''
        }
    )
    userForm = user_detailsForm(instance = user_for_det)


    if(req.method == 'POST'):
        userForm = user_detailsForm(req.POST , instance = user_for_det)
        if(userForm.is_valid()):
            userForm.save()
            return redirect('home')
        else:
            return redirect('user_details_form')
    context = {'userForm' : userForm}
    return render(req , 'user_details_register.html' , context)

def dashPage(req):
    if not req.user.is_superuser:
        return redirect('home')
    rev = Review.objects.all()
    choice = food.course_choice
    orders = order1.objects.all()
    order_items = orderedItems.objects.all()
    q = req.GET.get('q') if req.GET.get('q') != None else ''
    
    order_count = 0
    for order in orders:
        if order.status not in ["Delivered", "Canceled"]:
            order_count += 1


    if q == "Menu":
        item = food.objects.all()
    else:
        item = food.objects.filter(Q(course__icontains = q))
    context = {"food" : item , "course" : choice , "rev" : rev , "q" : q , "orders":orders , "order_items" : order_items , "order_count":order_count}
    return render(req , 'dashboard.html' , context)

def menuCreation(req):
    menuForm = foodForm()
    if(req.method == "POST"):
        menuForm = foodForm(req.POST , req.FILES)
        if(menuForm.is_valid()):
            menuForm.save()
        return redirect('dashboard')
    context = {'menuForm' : menuForm}
    return render(req , 'menuCRUD.html',context)

def menuUpdation(req , pk):
    item = food.objects.get(id = pk)
    menuForm = foodForm(instance = item)

    if(req.method == "POST"):
        menuForm = foodForm(req.POST , req.FILES , instance=item)
        if menuForm.is_valid():
            menuForm.save()
            return redirect("dashboard")
    context = {"menuForm" : menuForm}
    return render(req , 'menuCRUD.html' , context)

def menuDeletion(req , pk):
    item = food.objects.get(id = pk)
    if(req.method == "POST"):
        item.delete()
        return redirect('dashboard')
    return render(req , 'delete.html')


# def foodPage(req, pk):
#     items = get_object_or_404(food, id=pk)
#     rev = Review.objects.filter(food=items)

#     if req.method == "POST":
#         if not req.user.is_authenticated:
#             messages.warning(req, "Please login to add a review")
#             return redirect('login')

#         body = req.POST.get('body')
#         rating = req.POST.get('rating')

#         if not body or not rating:
#             messages.error(req, "Review and rating are required")
#         else:
#             Review.objects.create(
#                 user=req.user,
#                 food=items,
#                 body=body,
#                 rating=rating
#             )
#             messages.success(req, "Review added successfully")
#             return redirect('foodPage', pk=pk)

#     return render(req, 'food.html', {'items': items, 'rev': rev})

def foodPage(req, pk):
    items = get_object_or_404(food, id=pk)
    rev = Review.objects.filter(food=items)

    if req.method == "POST":
        if not req.user.is_authenticated:
            messages.warning(req, "Please login to add a review")
            return redirect('login')

        body = req.POST.get('body')

        try:
            rating = int(req.POST.get('rating'))
        except (TypeError, ValueError):
            messages.error(req, "Please select a valid rating")
            return redirect('foodPage', pk=pk)
        print("POST DATA:",req.POST)

        if not body:
            messages.error(req, "Review text is required")
        elif rating < 1 or rating > 5:
            messages.error(req, "Rating must be between 1 and 5")
        else:
            Review.objects.create(
                user=req.user,
                food=items,
                body=body,
                rating=rating
            )
            messages.success(req, "Review added successfully")
            return redirect('foodPage', pk=pk)

    return render(req, 'food.html', {'items': items, 'rev': rev})


@login_required(login_url='login')
def user_orders(req):
    orders = order1.objects.filter(user=req.user)
    orderItems = orderedItems.objects.filter(order__in=orders)

    q = req.GET.get('q')
    if q:
        try:
            canOrder = order1.objects.get(id=q, user=req.user)
            if canOrder.status == "Delivered":
                messages.warning(req, "Delivered orders cannot be canceled")
            else:
                canOrder.status = 'Canceled'
                canOrder.save()
                messages.success(req, "Order canceled successfully")
        except order1.DoesNotExist:
            messages.error(req, "Invalid order")

    return render(req, 'userOrders.html', {
        'orders': orders,
        'orderItems': orderItems
    })

# def create_admin_once(request):
#     if not User.objects.filter(username="admin").exists():
#         User.objects.create_superuser(
#             username="admin",
#             password="admin123",
#             email="admin@example.com"
#         )
#         return HttpResponse("Admin created successfully")
#     return HttpResponse("Admin already exists")
