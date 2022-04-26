# https://github.com/viscom-ulm/MCCNN/blob/master/utils/DataSet.py
import torch
import numpy as np
import time
from sklearn.neighbors import KDTree




class SampleMethod():
    def __init__(self, rate=0.8):
        self.rate = rate

        self.randomState_ = np.random.RandomState(int(time.time()))

    def _non_uniform_sampling_split_(self,  points, inNumPoints, inFeatures=None, numPoints=0, lowProbability=0.1):
        '''
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list.
            inFeatures (nxm np.array): List of point features.
            numPoints (int): Number of points to sample. If 0, all the points are selected.
            lowProbability (float): Probability used for the points in the second half of the bounding box.
            :return:
        '''
        shuffle_idx = np.random.choice(points.shape[0], points.shape[0], replace=False)
        points = points[shuffle_idx, :]
        coordMax = np.max(points, axis=0)
        coordMin = np.min(points, axis=0)
        aabbSize = coordMax - coordMin

        largestAxis = np.argmax(aabbSize)
        auxOutPts = []
        auxOutInFeatures = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            for i in range(inNumPoints):
                currPt = points[i]
                ptPos = (currPt[largestAxis] - coordMin[largestAxis]) / aabbSize[largestAxis]
                if ptPos > 0.5:
                    probVal = 1.0
                    rndNum = 0
                else:
                    probVal = lowProbability
                    rndNum = self.randomState_.random_sample()
                # Determine if we select the point.

                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(currPt)
                    if not (inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break
            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures],axis=1)

        return npOutPts


    def _non_uniform_sampling_lambert_(self, viewDir, points, normals, inNumPoints, inFeatures=None,
         numPoints=0):
        '''
        viewDir (3 np.array): View vector used to compute the probability of each point.
        points (nx3 np.array): List of points.
        normals (nx3 np.array): List of point normals.
        inNumPoints (int): Number of points in the list.
        inFeatures (nxm np.array): List of point features.
        numPoints (int): Number of points to sample. If 0, all the points are selected.
        :return:
        '''

        auxOutPts = []
        auxOutInFeatures = []

        exitVar = False
        numAddedPts = 0
        while not exitVar:
            for i in range(inNumPoints):
                # Compute the point probability.
                probVal = np.dot(viewDir, normals[i])
                probVal = pow(np.clip(probVal, 0.0, 1.0), 0.5)
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(points[i])
                    if not (inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break
            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts


    def _non_uniform_sampling_occlusion_(self, viewDir, points, normals, inNumPoints, inFeatures=None,
                                          numPoints=0, screenResolution=128):
        '''
        viewDir (3 np.array): View vector used to compute the visibility of each point.
        points (nx3 np.array): List of points.
        normals (nx3 np.array): List of point normals.
        inNumPoints (int): Number of points in the list.
        inFeatures (nxm np.array): List of point features.
        inLabels (nxl np.array): List of point labels.
        numPoints (int): Number of points to sample. If 0, all the points are selected.
        :return:
        '''
        xVec = np.cross(viewDir, np.array([0.0, 1.0, 0.0]))
        xVec = xVec / np.linalg.norm(xVec)
        yVec = np.cross(xVec, viewDir)
        yVec = yVec / np.linalg.norm(yVec)

        # Compute the bounding box.
        coordMax = np.max(points, axis=0)
        coordMin = np.min(points, axis=0)
        diagonal = np.linalg.norm(coordMax - coordMin) * 0.5
        center = (coordMax + coordMin) * 0.5

        # Create the screen pixels
        screenSize = screenResolution
        pixelSize = diagonal / (float(screenSize) * 0.5)
        screenPos = center - viewDir * diagonal - xVec * diagonal - yVec * diagonal
        screenZBuff = np.full([screenSize, screenSize], -1.0)
        # Compute the z value and pixel id in which each point is projected.
        pixelIds = [[-1, -1] for i in range(inNumPoints)]
        zVals = [1.0 for i in range(inNumPoints)]
        for i in range(inNumPoints):
            # If the point is facing the camera.
            if np.dot(normals[i], viewDir) < 0.0:
                # Compute the z value of the pixel.
                transPt = points[i] - screenPos
                transPt = np.array([
                    np.dot(transPt, xVec),
                    np.dot(transPt, yVec),
                    np.dot(transPt, viewDir) / (diagonal * 2.0)])
                zVals[i] = transPt[2]

                # Compute the pixel id in which the point is projected.
                transPt = transPt / pixelSize
                pixelIds[i][0] = int(np.floor(transPt[0]))
                pixelIds[i][1] = int(np.floor(transPt[1]))

                # Update the z-buffer.
                if screenZBuff[pixelIds[i][0]][pixelIds[i][1]] > zVals[i] or screenZBuff[pixelIds[i][0]][
                    pixelIds[i][1]] < 0.0:
                    screenZBuff[pixelIds[i][0]][pixelIds[i][1]] = zVals[i]

        auxOutPts = []
        auxOutInFeatures = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Determine if the point is occluded.
                if (zVals[i] - screenZBuff[pixelIds[i][0]][pixelIds[i][1]]) < 0.01:
                    # Store the point in the output buffers.
                    auxOutPts.append(points[i])
                    if not (inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])

                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts


    def _non_uniform_sampling_gradient_(self, points, inNumPoints, inFeatures=None, numPoints=0):
        """Method to non-uniformly sample a point cloud using the gradient protocol. The probability to select a
        point is based on its position alogn the largest axis of the bounding box.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list.
            inFeatures (nxm np.array): List of point features.
            inLabels (nxl np.array): List of point labels.
            numPoints (int): Number of points to sample. If 0, all the points are selected.

        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """
        # Compute the bounding box.
        shuffle_idx = np.random.choice(points.shape[0], points.shape[0], replace=False)
        points = points[shuffle_idx, :]
        coordMax = np.max(points, axis=0)
        coordMin = np.min(points, axis=0)
        aabbSize = coordMax - coordMin
        largestAxis = np.argmax(aabbSize)

        auxOutPts = []
        auxOutInFeatures = []
        exitVar = False
        numAddedPts = 0

        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Compute the point probability.
                currPt = points[i]
                probVal = (currPt[largestAxis]-coordMin[largestAxis]-aabbSize[largestAxis]*0.2
                    )/(aabbSize[largestAxis]*0.6)
                probVal = pow(np.clip(probVal, 0.01, 1.0), 1.0/2.0)
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(currPt)
                    if not(inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])

                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):

                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        if not (inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts


    def _uniform_sampling_(self, points, inFeatures=None, ):
        """Method to uniformly sample a point cloud.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list.
            inFeatures (nxm np.array): List of point features.
        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """

        npOutPts = np.array(points)
        if not (inFeatures is None):
            npOutInFeatures = np.array(inFeatures)
            npOutPts = np.concatenate([npOutPts, npOutInFeatures], axis=1)

        return npOutPts


    def _knn_non_uniform_sampling_(self, points, inFeatures, min_Points=100):
        key_point = points[0, :].reshape(-1,3)
        kdt = KDTree(points)
        keep_idx = kdt.query(key_point[:, :3], k=points.shape[0] // 2, return_distance=False)[0]
        keep_points = points[keep_idx, :]
        keep_features = inFeatures[keep_idx,:]

        need_sample_points = points[np.setdiff1d(np.arange(points.shape[0]), keep_idx), :]
        need_sample_points_features = inFeatures[np.setdiff1d(np.arange(points.shape[0]), keep_idx), :]

        if need_sample_points.shape[0] // 10 < min_Points - len(keep_idx):
            real_need_sample_point_num = min_Points - len(keep_idx)
        else:
            real_need_sample_point_num = need_sample_points.shape[0] // 10

        sample_idx = np.random.choice(np.arange(need_sample_points.shape[0]), real_need_sample_point_num)

        reshample_point = need_sample_points[sample_idx, :]
        resample_features = need_sample_points_features[sample_idx, :]
        final_point = np.concatenate([keep_points, reshample_point], 0)
        final_features = np.concatenate([keep_features, resample_features], 0)

        return np.concatenate([final_point, final_features],1)


    def random_sample(self, points, attribute=None, min_point=100):
        # random_strategy = [self._non_uniform_sampling_lambert_, self._non_uniform_sampling_occlusion_,
        #                    self._non_uniform_sampling_split_, self._uniform_sampling_]
        choice = self.randomState_.choice(4,1)[0]

        # print(choice)

        numPoints = int(self.rate * points.shape[0])
        if numPoints <= min_point:
            numPoints = min_point

        if choice==0:
            point_with_attribute = self._non_uniform_sampling_split_(points, len(points), attribute, numPoints=numPoints)
        elif choice==1:
            point_with_attribute = self._non_uniform_sampling_gradient_(points, len(points),attribute, numPoints=numPoints)
        elif choice==2:
            point_with_attribute = self._knn_non_uniform_sampling_(points, attribute, min_Points=min_point )
        #     auxView = (self.randomState_.rand(3) * 2.0) - 1.0
        #     auxView = auxView / np.linalg.norm(auxView)
        #     point_with_attribute = self._non_uniform_sampling_lambert_(auxView, points, inNumPoints=len(points),normals=attribute[:,:3],inFeatures=attribute, numPoints=numPoints)
        else:
            point_with_attribute = self._uniform_sampling_(points, attribute)

        if point_with_attribute.shape[0] < min_point:
            print(choice)
            print('original points shape:', points.shape)
            print('points with attribute shape:', point_with_attribute.shape)
            raise  ValueError
        assert  point_with_attribute.shape[0] >= min_point

        return point_with_attribute
