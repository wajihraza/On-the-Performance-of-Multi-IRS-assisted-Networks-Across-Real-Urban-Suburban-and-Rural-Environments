#!pip install gdal
#!apt-get install -y python3-gdal
#!pip install mayavi
#!pip install matplotlib mplleaflet

from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import multiprocessing as mp
from osgeo import ogr

#Assigning all values

c = 3*pow(10,8)
fc = 2.4*pow(10,9)
B = 20*10**6
a_L = 2
a_n = 3.3
x_uhf = 1
pt = 30
Pt_dB = 10*np.log10(pt) # transmitted power in dB
rate_threshold = 2e5
Pr_dB = np.array([])
u = 1
m = 4.0 #Shape for nakagami
omega = np.sqrt(0.5) #Param for nakagami
pr_los = []
pr_nlos = []
N1 = 1024
bs_coords = [(72.989, 33.6428, 35)]
bs = bs_coords[0]
irs_coords = [(72.992, 33.642, 20)]
irs = irs_coords[0]
n_bs = len(bs)
n_irs = len(irs)

#n_user = 200
N_val = [1024] # number of IRS Elements
NF = 10
mu = 0
std_dev = np.sqrt(0.5)
K = 3
user_Long_range = [72.983, 72.995]
user_Lat_range = [33.639, 33.647]


#FUNCTIONS
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # Radius of the Earth in kilometers
    radius = 6371.0
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # Calculate distance
    distance = radius * c
    return distance*1000

# Function to check LOS with obstacles
def has_los_with_obstacles_1(p1, p2, obstacles=[]):
    line = LineString([p1, p2])
    for obstacle in obstacles:
        if line.intersects(obstacle):
            return False
    return True

# Function to check LOS with obstacles and building heights using grid-based approach
def has_los_with_obstacles(p1, p2, obstacles=[], grid_resolution=50):
    x_vals = np.linspace(p1[0], p2[0], grid_resolution)
    y_vals = np.linspace(p1[1], p2[1], grid_resolution)
    z_vals = np.linspace(p1[2], p2[2], grid_resolution)

    for i in range(grid_resolution):
        point = (x_vals[i], y_vals[i], z_vals[i])

        for obstacle in obstacles:
            if obstacle.contains(Point(point[:2])):
                obstacle_row = obstacle_gdf[obstacle_gdf['geometry'] == obstacle]
                obstacle_height = obstacle_row.iloc[0]['height']
                if point[2] <= obstacle_height:
                    return False
                break  # No need to check other obstacles if one is found
    return True

def compute_pathloss(distance,path_loss_exponent):
  y = c/fc
  path_loss = 20*np.log10(4*np.pi/y)+10*path_loss_exponent*np.log10(distance)+x_uhf
  path_loss = np.array(path_loss)
  return path_loss

def generate_nakagami_samples(m, omega, size):
    magnitude_samples = np.sqrt(omega) * np.sqrt(np.random.gamma(m, 1, size)) / np.sqrt(np.random.gamma(m - 0.5, 1, size))
    phase_samples = np.random.uniform(0, 2 * np.pi, size=size)
    complex_samples = magnitude_samples * np.exp(1j * phase_samples)
    return complex_samples

def compute_distances(user_positions, base_stations):
    distances = np.sqrt(np.sum(np.square(user_positions - base_stations), axis=1))
    return distances

def generate_rayleigh_fading_channel(num_users, std_mean, std_dev):
    X = np.random.normal(std_mean, std_dev, num_users)
    Y = np.random.normal(std_mean, std_dev, num_users)
    rayleigh_channel = np.abs(X + 1j*Y)
    return rayleigh_channel

def rician_fading_channel(num_samples, K, sigma):
    gaussian_samples = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
    # Generate Rician fading channel samples
    h = np.sqrt(K / (K + 1)) * (gaussian_samples + np.sqrt(1 / (2 * (K + 1))) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)))
    noise = sigma * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    h_with_noise = h + noise
    return abs(h_with_noise)

# Function to compute outage probability at each iteration
def compute_outage_probability(num_users, rate, rate_threshold):
    outage = 0
    for j in range(num_users):
        if rate[j] < rate_threshold:
            outage += 1
    return outage / num_users

def compute_SNR(PR, noise_floor):
    SNR = PR - noise_floor
    #SNR = (10 ** (SNR/10))
    return SNR

def compute_rate(SNR):
    SNR_watts = (10**(SNR/10))
    
    if len(SNR) == 0:
        x = 1
    else:
        x = len(SNR)
    return (B/(x))*np.log2(1 + SNR_watts)

def total_coverage_probability(rate_mat, threshold_rate):
    coverage_probability = np.sum(rate_mat > threshold_rate) / n_user
    return coverage_probability

def find_common_locations(locations1, locations2):
    common_locations = []
    for loc1 in locations1:
        for loc2 in locations2:
            if not np.array_equal(loc1, [0, 0]) and not np.array_equal(loc2, [0, 0]) and np.all(loc1 == loc2):
                common_locations.append(tuple(loc1))
                break  # Once found, no need to continue checking
    return common_locations

def replace_common_locations(channel, common):
    new_channel = []
    for loc in channel:
        if tuple(loc) in common:
            new_channel.append([0, 0])
        else:
            new_channel.append(loc)
    return new_channel

def coverage_probability(rate_mat, threshold_rate, n_user):
    coverage_probability = np.sum(rate_mat > threshold_rate) / n_user
    return coverage_probability


# Iterate through each base station
# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Users\\ibrah\\Downloads\\FYP_CODE\\output_shapefile_3.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)
# Define base station and user coordinates
#user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
#user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
#user_coords = list(zip(user_longitudes, user_latitudes))

# Create empty lists for LOS and NLOS users
all_los_users_x = []
all_los_users_y = []
all_nlos_users_x = []
all_nlos_users_y = []


los_users = []
nlos_users = []



# Scatter plot for base stations
bs_longitudes = [coord[0] for coord in bs_coords]
bs_latitudes = [coord[1] for coord in bs_coords]

# Arrays to store LOS and NLOS users for each BS
los_users_per_bs = [[] for _ in range(n_bs)]
nlos_users_per_bs = [[] for _ in range(n_bs)]


# Create empty lists for user markers
los_users = []          # LOS User from bs
irs_los_users = []      # IRS-LOS User
irs_nlos_users = []     # IRS-NLOS User
other_users = []        # Remaining Users

# Iterate through each user
#for user_coord in user_coords:
#    user_point = Point(user_coord)
#    los_flag = False
#    irs_los_flag = False
#    irs_nlos_flag = False

    # Check LOS between each base station and user

# Add random heights to the GeoDataFrame
random_obstacle_heights = np.random.uniform(10, 17, len(obstacle_gdf))
obstacle_gdf['height'] = random_obstacle_heights

#user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
#user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
#user_heights = np.random.uniform(0,0.5, n_user)  # Random user heights
#user_coords = list(zip(user_longitudes, user_latitudes, user_heights))

# Create empty lists for user markers
los_users = []          # LOS User from bs
irs_los_users = []      # IRS-LOS User
irs_nlos_users = []     # IRS-NLOS User
other_users = []        # Remaining Users
has_los_bs_irs = []

n_user_vals = np.linspace(30,350,150).astype(int)  # Values of n_user

rate_per_unit_area = np.zeros((len(n_user_vals)),dtype=object)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def process_coverage2(n_user_idx,  n_user, n_iterations,q):

#def process_coverage1(N_idx, N1, n_user_idx,  n_user, n_iterations,q):
    
    snr = []
    final_rate = []
    N1 = 1024
    for _ in range(n_iterations):
        # Rest of the code remains the same as before
        user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1],int(n_user))
        user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], int(n_user))
        user_heights = np.random.uniform(0,0.5, n_user)  # Random user heights
        user_coords = list(zip(user_longitudes, user_latitudes, user_heights))

        los_users = []
        nlos_users = []
        irs_los_users = []
        irs_nlos_users = []
        
        for user_coord in user_coords:
            user_point = Point(user_coord[:2])
            user_height = user_coord[2]
            has_los = has_los_with_obstacles((bs[0], bs[1], bs[2]), (user_point.x, user_point.y, user_height), obstacle_gdf['geometry'])
            if has_los:
                los_users.append(user_coord)
            else:
                nlos_users.append(user_coord)
                # Check LOS between IRS and NLOS users if BS-IRS link exists
                has_los_bs_irs = True
                if has_los_bs_irs:
                    has_los_irs_user = has_los_with_obstacles((irs[0], irs[1], irs[2]), (user_point.x, user_point.y, user_height), obstacle_gdf['geometry'])
                    if has_los_irs_user:
                        irs_los_users.append(user_coord)
                    else:
                        irs_nlos_users.append(user_coord)
        los_users = np.array(los_users)
        irs_los_users = np.array( irs_los_users)
        irs_nlos_users = np.array( irs_nlos_users)
        dist_irs_los_users = []
        dist_bs_irs = []
        dist_irs_nlos_users = []
        #Distance for directly los users from base station to users
        # Instead of accessing los_users[:, 1] and los_users[:, 0], use the following
        dist_los_users = haversine_distance([user[1] for user in los_users], [user[0] for user in los_users], bs[1], bs[0])
        dist_los_users = np.sqrt(dist_los_users**2+(los_users[:,2]-bs[2])**2)

        dist_irs_nlos_users = haversine_distance([user[1] for user in irs_nlos_users], [user[0] for user in irs_nlos_users], bs[1], bs[0])
        dist_irs_nlos_users = np.sqrt(dist_irs_nlos_users**2+(irs_nlos_users[:,2]-bs[2])**2)


        # Calculate distance between users and base station for IRS-LOS users that are NLOS from the base station directly
        dist_irs_los_users = [haversine_distance(user[1], user[0], bs[1], bs[0]) for user in irs_los_users]
        dist_irs_los_users = np.sqrt((np.array(dist_irs_los_users)**2) + (np.array([user[2] for user in irs_los_users]) - bs[2]) ** 2)

        #Distance between base station and IRS
        dist_bs_irs = haversine_distance(irs[1],irs[0], bs[1], bs[0])
        dist_bs_irs = np.sqrt(dist_bs_irs**2+(irs[2]- bs[2])**2)
        
        #Distance between
        irs_distance = [haversine_distance(user[1], user[0], irs[1], irs[0]) for user in irs_los_users]
        irs_distance = np.sqrt(np.array(irs_distance)**2 + (np.array([user[2] for user in irs_los_users]) - irs[2]) ** 2)

        if len(irs_los_users) == 0:
            N = 0
        else:
            N = int(N1/len(irs_los_users))

        # Rayleigh Fading Channel for los user
        g_m = rician_fading_channel(len(los_users), K, omega)
        g_m = g_m.reshape(1,-1)
        los_dist = dist_los_users
        
        los_pathloss = compute_pathloss(los_dist,a_L)
        los_pathloss = los_pathloss.reshape(1,-1)
        
        Pr_los = []
        for pl_los in los_pathloss:
            Pr_los = np.append(Pr_los, Pt_dB + 10*np.log10(g_m**2)- pl_los ) # received power for LOS links
        Pr_los = Pr_los.reshape(1,-1)
        
        c_m = generate_rayleigh_fading_channel(len(irs_nlos_users), mu, std_dev)
        c_m = c_m.reshape(1,-1)
        nlos_dist = dist_irs_nlos_users
        nlos_pathloss = compute_pathloss(nlos_dist,a_n)
        nlos_pathloss = nlos_pathloss.reshape(1,-1)
        Pr_nlos = []
        for pl_nlos in nlos_pathloss:
            Pr_nlos = np.append(Pr_nlos, Pt_dB + 10*np.log10(c_m**2)- pl_nlos ) # received power for LOS links
        Pr_nlos = Pr_nlos.reshape(1,-1)

        # Distances from each user to the Base Station
        d_m = dist_irs_los_users
        d_m = np.squeeze(d_m)
        d_m_PL = compute_pathloss(d_m,a_n)
        d_m_PL = 10**(d_m_PL/10)
        d_m = d_m_PL.reshape(1,len(irs_los_users))

        # Distances from each user to the IRS
        d_rm =  irs_distance
        d_rm = np.squeeze(d_rm)
        # Distances from Base Station to the IRS
        d_i = dist_bs_irs
        d_i = d_i.reshape(-1)
        p_di = compute_pathloss(d_rm*d_i,a_L)
        p_di = p_di.reshape(1,-1)
        p_di = 10**(p_di/10)

        # Rayleigh Fading Channel for nlos user for direct path
        h_m = generate_rayleigh_fading_channel(len(irs_los_users), mu, std_dev)
        h_m = h_m.reshape(1,-1)
        f_m = []
        fading = []
        for user in range(len(irs_los_users)):
            fading.append(generate_nakagami_samples(m, omega, N))
        f_m = np.array(fading)
        f_m = f_m.reshape(N,len(irs_los_users))
        f_m_transpose = np.transpose(f_m)
        # Generate the Nakagami Channel from base_station to the IRS
        g = generate_nakagami_samples(m, omega, N)
        g = g.reshape(N,1)

        r = []
        for row_index in range(len(irs_los_users)):
            single_row = f_m_transpose[row_index,:]
            result = np.dot(single_row, g)
            r.append(result)
        r = np.squeeze(r)
        r = r.reshape(1,len(irs_los_users))
        product = abs(r)
        PR = []
        for i in range(len(irs_los_users)):

            PR_W = (pt) * (h_m[0][i]**2)/(d_m[0][i]) + pt*(product[0][i]**2)/(p_di[0][i])
            PR.append(PR_W)
       
        # Convert the list to a numpy array
        PR = np.array(PR)
        PR = PR.reshape(1,len(irs_los_users))

        Pr_los_irs = 10 * np.log10(PR)
        Pr_los_irs = Pr_los_irs.reshape(1,len(irs_los_users))
        Pr_dB1 = np.concatenate((Pr_los, Pr_los_irs),axis = None)
        Pr_dB = np.concatenate((Pr_nlos, Pr_dB1),axis = None)

        np.random.shuffle(Pr_dB)
        Pr_dB = Pr_dB.reshape(-1)
        
        #Calculate the noise value
        noise = -174+10*np.log10(B) + NF
        SNR_matrix = []

        # Calculate SNR
        for i in range(n_user):
            SNR = compute_SNR(Pr_dB[i], noise)
            SNR_matrix.append(SNR)

        SNR_matrix = np.array(SNR_matrix)
        SNR_matrix = SNR_matrix.reshape(1,n_user)

        rate_matrix = compute_rate(SNR_matrix)
        rate_matrix = rate_matrix.reshape(-1)
        final_rate.append(rate_matrix)
    rate = np.mean(final_rate, axis=0)
    rate = np.squeeze(rate)    
    q.put(rate)
    
    snr = np.mean(SNR_matrix, axis=0)
    snr = np.squeeze(snr)
    #q.put(snr)

      
# Create a pool of processes

if __name__ == "__main__":

    # Define simulation parameters
    n_iterations = 100  # Number of iterations
    n_user_vals = np.linspace(30,350,150).astype(int)  # Values of n_user

    # Threshold rate for coverage probability calculation

    for n_user_idx, n_user in enumerate(n_user_vals):
        print(n_user_idx)
        q1 = mp.Queue()
        q2 = mp.Queue()
        q3 = mp.Queue()
        q4 = mp.Queue()
        q5 = mp.Queue()
        q5 = mp.Queue()
        q6 = mp.Queue()
        q7 = mp.Queue()
        q8 = mp.Queue()
        q9 = mp.Queue()
        q10 = mp.Queue()

        p1 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q1))
        p2 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q2))
        p3 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q3))
        p4 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q4))
        p5 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q5))
        p6 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q6))
        p7 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q7))
        p8 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q8))
        p9 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q9))
        p10 = mp.Process(target=process_coverage2, args=(n_user_idx,  n_user, n_iterations,q10))
    
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        p9.start()
        p10.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()
        p9.join()
        p10.join()
        
        rate_per_unit_area[n_user_idx] = (q1.get() +q2.get()+q3.get()+q4.get()+q5.get()+q6.get()+q7.get()+q8.get()+q9.get()+q10.get())/10

    print("Output Shape: ", np.shape(rate_per_unit_area))
    print("Output is ", rate_per_unit_area)
    np.save('code_3_1_irs.npy',rate_per_unit_area)

