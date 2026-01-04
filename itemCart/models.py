from django.db import models
from rbc.models import food , user_details
from django.contrib.auth.models import User

# Create your models here.

class cartSes(models.Model):
    user = models.ForeignKey(User , on_delete=models.CASCADE)
    items = models.ManyToManyField(food)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Cart {self.id}"

class cartItems(models.Model):
    cart = models.ForeignKey(cartSes, on_delete=models.CASCADE)
    item = models.ForeignKey(food , on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)

    class Meta:
        unique_together = ('cart' , 'item')

    def __str__(self):
        return f"{self.cart}   {self.item}    {self.quantity}"
    

class order1(models.Model):
    class stat(models.TextChoices):
        queue = "In Queue"
        making = "Preparing!"
        completed = "Prepared."
        out = "Out For Delevery"
        done = "Delivered"
        canceled = "Canceled"
    
    user = models.ForeignKey(User , on_delete=models.CASCADE)
    user_det = models.ForeignKey(user_details , on_delete=models.CASCADE)
    cart = models.ForeignKey(cartSes, on_delete=models.CASCADE)
    status = models.CharField(max_length=17 , choices=stat , default=stat.queue)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f'Order {self.id}'

class orderedItems(models.Model):
    order = models.ForeignKey(order1, on_delete=models.CASCADE , default=1)
    item = models.ForeignKey(food , on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)

    def __str__(self):
        return f"{self.order} : {self.quantity} x {self.item}"

class Contact(models.Model):
    contact=models.CharField(max_length=12)

    def __str__(self):
        return self.contact


class Address(models.Model):
    address=models.CharField(max_length=200)
    pin=models.CharField(max_length=10,default="000000")
    direction=models.URLField()

    def __str__(self):
        return self.address