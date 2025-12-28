from django.shortcuts import render , get_object_or_404 , redirect
from rbc.models import food , user_details
from .models import cartItems , cartSes , order1 , orderedItems
from django.contrib.auth.models import User
from django.http import JsonResponse
from rbc import views
from .forms import cartForm
from django.forms import modelformset_factory
from django.contrib.auth.decorators import login_required

from django.db.models import Q

# Create your views here.


def cartPage(req):
    page = 'cart'
    cart = cartSes.objects.get(user = req.user)
    cartItem = cartItems.objects.filter(cart = cart)
    
    amount = 0
    for i in cartItem:
        amount += (i.item.price * i.quantity)
    context = {'cart' : cart , 'cartItems' : cartItem , 'amount' : amount , 'page' : page}

    return render(req, 'cart.html' , context)

@login_required(login_url='login')
def cartAdd(req):
    cart = cartSes.objects.get(user = req.user)

    addCart = req.GET.get('q')

    item = get_object_or_404(food, id=addCart)

    if not cartItems.objects.filter(cart=cart, item=item).exists():
        cartItems.objects.create(cart=cart, item=item)

    return redirect('home')

def cartUpdate(req):
    page = 'cart'
    cartFormSet = modelformset_factory(cartItems , form = cartForm, extra=0)
    cart = cartSes.objects.get(user = req.user)
    cartItem = cartItems.objects.filter(cart = cart)
    forms = cartFormSet(queryset = cartItem)

    if(req.method == 'POST'):
        forms = cartFormSet(req.POST)
        print(forms.is_valid())
        if forms.is_valid():
            forms.save()
            return redirect('cartPage')
        else:
            print(forms.errors)

    context = {'forms' : forms , 'cartItem' : cartItem , 'page' : page}
    return render(req , 'updateCart.html' , context)

def cartDelete(req):
    itemId = req.GET.get('q')
    item = cartItems.objects.get(id = itemId)
    item.delete()
    return redirect('cartPage')


def orderPlace(req):
    cart = cartSes.objects.get(user = req.user)
    cartItem = cartItems.objects.filter(cart = cart)
    user_data = user_details.objects.get(user = req.user)

    order = order1.objects.create(
        user = req.user,
        user_det = user_data,
        cart = cart,
    )
    print(order)
    for item in cartItem:
        orderedItems.objects.create(
            order = order,
            item = item.item,
            quantity = item.quantity
        )
    for item in cartItem:
        item.delete()

    return redirect('home')
    