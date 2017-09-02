function dist = getEucledianDist(pointA, pointB)

diff = pointA - pointB;
sqDiff = diff .^2;
dist = sum(sqDiff);
end
