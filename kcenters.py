import numpy 
from random import randint
from timeit import default_timer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score

class KCenters:
    def __init__(self, S, c, p):
        self.S = S
        self.c = c
        self.p = p
        self.maxR = numpy.inf
        self.distances = numpy.zeros((len(S), len(S)))
        for i in range(len(S)):
            self.distances[i, :] = self.getMinkowskiDistance(i, S, S)
            

    def getMinkowskiDistance(self, i, S, C):
        return numpy.sum(numpy.abs(S[i] - C) ** self.p, axis=1) ** (1/self.p)


    def getGreedyKCenters(self):
        if self.c > len(self.S):
            return self.S

        centers = []
        distances = numpy.full(len(self.S), numpy.inf)
        maxDistanceIdx = randint(0, len(self.S) - 1)

        aux = self.c

        while (aux):
            centers.append(maxDistanceIdx)

            for i in range(len(self.S)):
                distances[i] = min(distances[i], self.distances[i][maxDistanceIdx])

            maxDistanceIdx = numpy.argmax(distances)
            aux -= 1

        self.maxR = distances[maxDistanceIdx]

        return centers
    

    def getLabels(self):
        centers = None

        if self.c > len(self.S):
            centers = self.S
        else:
            centers = []
            distances = numpy.full(len(self.S), numpy.inf)
            maxDistanceIdx = randint(0, len(self.S) - 1)

            aux = self.c

            while (aux):
                centers.append(maxDistanceIdx)

                for i in range(len(self.S)):
                    distances[i] = min(distances[i], self.distances[i][maxDistanceIdx])

                maxDistanceIdx = numpy.argmax(distances)
                aux -= 1

            self.maxR = distances[maxDistanceIdx]

        labels = numpy.full(len(self.S), -1)

        for i in range(len(self.S)):
            for j in range(self.c):
                if labels[i] == -1:
                    labels[i] = j

                elif self.distances[i][centers[j]] <= self.distances[i][centers[labels[i]]]:
                    labels[i] = j

        return labels


    def calculateResults(self):
        labels = self.S[:, self.c]
        S = numpy.delete(self.S, self.c, axis=1)

        k = len(numpy.unique(labels))

        self = KCenters(S, k, self.p)
        self.distances = numpy.zeros((len(S), len(S)))
        
        for i in range(len(S)):
            self.distances[i, :] = self.getMinkowskiDistance(i, S, S)

        silhouette = numpy.zeros(30)
        rand = numpy.zeros(30)
        time = numpy.zeros(30)
        radius = numpy.zeros(30)  

        for i in range(30):
            start = default_timer()
            auxLabels = self.getLabels()
            end = default_timer()

            time[i] = end - start
            silhouette[i] = silhouette_score(self.distances, auxLabels, metric="precomputed")
            rand[i] = rand_score(labels, auxLabels)
            radius[i] = self.maxR            

        start = default_timer()
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(S)
        kmeansLabels = kmeans.labels_
        end = default_timer()
        
        maxDistance = 0
        for i in numpy.unique(labels):
            group = numpy.argwhere(labels == i)
            for j in group:
                for k in group:
                    distance = self.getMinkowskiDistance(j, self.S, self.S[k])
                    maxDistance = max(maxDistance, distance)
        optmizedRadius = maxDistance/2
        
        distances = numpy.zeros((len(self.S), len(kmeans.cluster_centers_)))
        for i in range(len(self.S)):
            distances[i, :] = self.getMinkowskiDistance(i, self.S, kmeans.cluster_centers_)
        maxRadius = numpy.max(numpy.min(distances, axis=1))

        result = "Algoritmo 2-Aproximado\n"
        result += " Medias\n"
        result += " - Tempo: %lf\n" % numpy.average(time)
        result += " - Silhueta: %lf\n" % numpy.average(silhouette)
        result += " - Indice de rand: %lf\n" % numpy.average(rand)
        result += " - Raio maximo: %lf\n" % numpy.average(radius)
        result += " Desvios padrao\n"
        result += " - Tempo: %lf\n" % numpy.std(time)
        result += " - Silhueta: %lf\n" % numpy.std(silhouette)
        result += " - Indice de rand: %lf\n" % numpy.std(rand)
        result += " - Raio maximo: %lf\n" % numpy.std(radius)
        result += "\nAlgoritmo K-Means\n"
        result += " - Tempo: %lf\n" % (end - start)
        result += " - Silhueta: %lf\n" % silhouette_score(self.distances, kmeansLabels, metric="precomputed")
        result += " - Indice de rand: %lf\n" % rand_score(labels, kmeansLabels)
        result += " - Raio maximo: %lf\n" % maxRadius
        result += "\nRaio verdadeiro: %lf\n" % optmizedRadius

        return result
        
