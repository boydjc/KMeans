import pandas as pd
import matplotlib.pyplot as plt
import random
from math import sqrt

class KMeans():

	def __init__(self):
		self.CLUSTERNUM = 3
		self.x = []
		self.minX = None
		self.maxX = None
		self.y = []
		self.minY = None
		self.maxY = None
		self.data = None
		self.SSE = 0
		self.finalCentroidPoints = None
		self.clusters = {}
	
		# initialize the cluster dict
		for count in range(0, self.CLUSTERNUM):
			self.clusters.update({'cluster' + str(count) : {'points' : []}})	

		self.previousCentroids = []
		self.centroids = []

	def loadData(self):
		# load the data
		with open('A.txt') as f:
			lines = f.readlines()
			for line in lines:
				lineSplit = line.split()
				self.x.append(float(lineSplit[0]))
				self.y.append(float(lineSplit[1]))

	def initializeCentroids(self):
		for count in range(0, self.CLUSTERNUM):
			centX = random.uniform(0, self.maxX)
			centY = random.uniform(0, self.maxY)
			self.centroids.append([centX, centY])

	def calculateDistortion(self):
		totalDistortion = 0
		for clusterCount in range(0, self.CLUSTERNUM):
			clusterDifference = 0
			for pointCount in range(0, len(self.clusters['cluster' + str(clusterCount)]['points'])):
				differenceX = (self.clusters['cluster' + str(clusterCount)]['points'][pointCount][0] - self.centroids[clusterCount][0])**2
				differenceY = (self.clusters['cluster' + str(clusterCount)]['points'][pointCount][1] - self.centroids[clusterCount][1])**2
				clusterDifference += differenceX + differenceY

			self.SSE += clusterDifference	

	def calculateCentroids(self):	
		# store the old centroid points
		self.previousCentroids.clear()

		for count in range(0, len(self.centroids)):
			self.previousCentroids.append(self.centroids[count])

		self.centroids.clear()

		for clusterCount in range(0, self.CLUSTERNUM):
			if not(len(self.clusters['cluster' + str(clusterCount)]['points']) == 0):
				newCentroidX = 0
				newCentroidY = 0
				for pointCount in range(0, len(self.clusters['cluster' + str(clusterCount)]['points'])):
					newCentroidX += self.clusters['cluster' + str(clusterCount)]['points'][pointCount][0]
					newCentroidY += self.clusters['cluster' + str(clusterCount)]['points'][pointCount][1]
		
				newCentroidX /= len(self.clusters['cluster' + str(clusterCount)]['points'])
				newCentroidY /= len(self.clusters['cluster' + str(clusterCount)]['points'])

				# store the new centroid
				self.centroids.append([newCentroidX, newCentroidY])
			else:
				# if we have a divide by 0 situation, assign the previous centroid values to 
				# this new centroid and try again next iteration
				self.centroids.append([self.previousCentroids[clusterCount][0], self.previousCentroids[clusterCount][1]])

	def assignPoints(self):
		# for each centroid, go through the list of points and 
		# which ever point is closest to that centroid assign it to that centroid
		# in the dictionary

		# clear the cluster points
		for count in range(0, self.CLUSTERNUM):
			self.clusters.update({'cluster' + str(count) : {'points' : []}})	

		for pointCount in range(0, len(self.x)):
			# calculate the distances
			centDistances = []
			for centCount in range(0, self.CLUSTERNUM):
				# difference between two points formula
				distance = sqrt((self.data['X'][pointCount] - self.centroids[centCount][0])**2 +
								(self.data['Y'][pointCount] - self.centroids[centCount][1])**2)

				centDistances.append(distance)
		
			minDistance = min(centDistances)
			minDistanceCluster = centDistances.index(minDistance)
			minDistanceClusterCentroidX = self.centroids[minDistanceCluster][0]
			minDistanceClusterCentroidY = self.centroids[minDistanceCluster][1]

			# assign this point to the corresponding cluster in the cluster dict
			clusterPoints = self.clusters['cluster' + str(minDistanceCluster)]['points']
			clusterPoints.append([self.data['X'][pointCount], self.data['Y'][pointCount]])
			self.clusters.update({'cluster' + str(minDistanceCluster) : {'points' : clusterPoints}})


	def performClustering(self):

		self.loadData()

		# make the dataframe with the data
		self.data = pd.DataFrame(list(zip(self.x, self.y)), columns=['X', 'Y'])

		# get the max and min values from the X and Y columns 
		# and use that to randomly generate the 3 centroids
		self.maxX = self.data['X'].max()
		self.minX = self.data['X'].min()

		self.maxY = self.data['Y'].max()
		self.minY = self.data['Y'].min()

		# initialize the centroids
		self.initializeCentroids()

		runCount = 0
		while not(self.previousCentroids == self.centroids): 
			# assign the points to each centroid
			self.assignPoints()

			# recalculate the centroids
			self.calculateCentroids()
			runCount+= 1


		# consolidate final centroid points
		self.finalCentroidPoints = pd.DataFrame(self.centroids, columns=['CentX', 'CentY'])

		plt.figure(figsize=(10, 7))
		plt.title('Data')

		# split the three clusters into dataframes to plot
		for count in range(0, self.CLUSTERNUM):
			plt.scatter(pd.DataFrame(self.clusters['cluster' + str(count)]['points'], columns=['X', 'Y'])['X'],
						pd.DataFrame(self.clusters['cluster' + str(count)]['points'], columns=['X', 'Y'])['Y'])
		
		# plot the centroid points
		plt.scatter(self.finalCentroidPoints['CentX'], self.finalCentroidPoints['CentY'], color='red', marker='x', s=100)

		plt.show()

		self.calculateDistortion()
		print('Distortion Function Value: ', self.SSE)


if __name__ == '__main__':
	kM = KMeans()
	kM.performClustering()
