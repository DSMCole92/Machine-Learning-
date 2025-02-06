if step %2 ==  0:
    run = run + 1
    print("Run " + str(run) + ": New Centers")
    if step == 0:
        sel = np.array([43,47,54])
        centers = pts[sel,:]
        track0[0].append(centers[0,0])
        track0[1].append(centers[0,1])
        track1[0].append(centers[1,0])
        track1[1].append(centers[1,1])
        track2[0].append(centers[2,0])
        track2[1].append(centers[2,1])
        clusters = np.zeros(len(pts))
    else: 
        centers = newCenters(pts, clusters, K)
        track0[0].append(centers[0,0])
        track0[1].append(centers[0,1])
        track1[0].append(centers[1,0])
        track1[1].append(centers[1,1])
        track2[0].append(centers[2,0])
        track2[1].append(centers[2,1])
else:
    print("Run " + str(run) + ": Assign Clusters")
    clusters = assign(pts,centers) 
step = step + 1

fig=plt.figure(figsize=(6,6), dpi= 80)
plt.scatter(pts[:,0], pts[:,1], c = clusters)
plt.scatter(centers[:,0], centers[:,1], c = 'Red')
plt.plot(track0[0],track0[1], c = 'Red')
plt.plot(track1[0],track1[1], c = 'Red')
plt.plot(track2[0],track2[1], c = 'Red')
plt.xlim([15,50])
plt.ylim([15,50])
plt.show()