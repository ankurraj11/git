from django.db import models

class collection(models.Model):
	genre = models.CharField(max_length = 50)
	category = models.CharField(max_length = 50)

	def __str__(self):
		return(self.genre + ' '+ '&' + ' '+ self.category)

class stories(models.Model):
	genre = models.CharField(max_length = 50)
	collection = models.ForeignKey(collection, on_delete = models.CASCADE)
	writer = models.CharField(max_length = 50)