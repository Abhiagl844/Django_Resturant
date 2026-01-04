from django.contrib import admin
from .models import cartItems , cartSes , order1 , orderedItems,Contact,Address

# Register your models here.

admin.site.register(cartSes)
admin.site.register(cartItems)
admin.site.register(order1)
admin.site.register(orderedItems)
admin.site.register(Contact)
admin.site.register(Address)