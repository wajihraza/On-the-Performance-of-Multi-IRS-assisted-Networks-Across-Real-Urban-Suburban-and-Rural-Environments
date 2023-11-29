from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import multiprocessing as mp
from osgeo import ogr
import pandas as pd
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
m = 1.0 #Shape for nakagami
omega = np.sqrt(0.5) #Param for nakagami
pr_los = []
pr_nlos = []
N_val = [512] # number of IRS Elements
N1 = N_val[0]
#For Monte Carlo Simulation
# Define base station and user coordinates
bs_coords = [(-87.36, 41.8835,150), (-87.826, 41.881,150), (-87.636, 41.884,150)]
 # List of base station coordinates
#bs_coords = [(72.989, 33.6428,70)]
# Define IRS coordinates
irs_coords = [(-87.623, 41.884,100), (-87.627, 41.885,100), (-87.633, 41.881,100)]  # List of IRS coordinates
#irs_coords = [(72.985, 33.645,10)]
n_bs = len(bs_coords)
n_irs = len(irs_coords)
bs =  bs_coords
irs = irs_coords
n_user = 400

NF = 10
mu = 0
std_dev = np.sqrt(0.5)
K = 4
user_Long_range = [-87.636, -87.622]
user_Lat_range = [41.880, 41.887]




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
    return SNR

def compute_rate(SNR):
    SNR_watts = (10**(SNR/10))
    
    if len(SNR) == 0:
        x = 1
    else:
        x = len(SNR)
    return (B/(n_bs*x))*np.log2(1 + SNR_watts)

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



# Load the shapefile into a GeoDataFrame
shapefile_path = "C:\\Visual Studio code\\Test\\output_shapefile_chicago.shp"
gdf = gpd.read_file(shapefile_path)


# Iterate through each base station
# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Visual Studio code\\Test\\output_shapefile_chicago.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)
n_bs = len(bs_coords)
n_irs = len(irs_coords)
# Define base station and user coordinates
user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
user_coords = list(zip(user_longitudes, user_latitudes))

# Create empty lists for LOS and NLOS users
all_los_users_x = []
all_los_users_y = []
all_nlos_users_x = []
all_nlos_users_y = []

# Iterate through each base station
for bs_coord in bs_coords:
    los_users = []
    nlos_users = []

    for user_coord in user_coords:
        user_point = Point(user_coord)
        has_los = has_los_with_obstacles_1((bs_coord[0], bs_coord[1]), user_point, obstacle_gdf['geometry'])
        if has_los:
            los_users.append(user_coord)
        else:
            nlos_users.append(user_coord)

    # Append LOS and NLOS user coordinates to all lists
    all_los_users_x.extend([coord[0] for coord in los_users])
    all_los_users_y.extend([coord[1] for coord in los_users])
    all_nlos_users_x.extend([coord[0] for coord in nlos_users])
    all_nlos_users_y.extend([coord[1] for coord in nlos_users])


# Scatter plot for base stations
bs_longitudes = [coord[0] for coord in bs_coords]
bs_latitudes = [coord[1] for coord in bs_coords]

# Arrays to store LOS and NLOS users for each BS
los_users_per_bs = [[] for _ in range(n_bs)]
nlos_users_per_bs = [[] for _ in range(n_bs)]

# Iterate through each user and check LOS from each BS
for user_coord in user_coords:
    user_point = Point(user_coord)
    for bs_index, bs_coord in enumerate(bs_coords):
        has_los = has_los_with_obstacles_1(bs_coord[:2], user_point, obstacle_gdf['geometry'])
        if has_los:
            los_users_per_bs[bs_index].append(user_coord)
        else:
            nlos_users_per_bs[bs_index].append(user_coord)


# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Visual Studio code\\Test\\output_shapefile_chicago.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)


# Create empty lists for user markers
los_users = []          # LOS User from bs
irs_los_users = []      # IRS-LOS User
irs_nlos_users = []     # IRS-NLOS User
other_users = []        # Remaining Users

# Iterate through each user
for user_coord in user_coords:
    user_point = Point(user_coord)
    los_flag = False
    irs_los_flag = False
    irs_nlos_flag = False

    # Check LOS between each base station and user
    for bs_coord in bs_coords:
        has_los = has_los_with_obstacles_1((bs_coord[0], bs_coord[1]), user_point, obstacle_gdf['geometry'])
        if has_los:
            los_flag = True
            break

    # Check LOS between IRS and user
    for irs_coord in irs_coords:
        has_los_irs_user = has_los_with_obstacles_1((irs_coord[0], irs_coord[1]), user_point, obstacle_gdf['geometry'])
        if has_los_irs_user:
            irs_los_flag = True
        else:
            irs_nlos_flag = True

    # Assign user to appropriate marker list
    if los_flag:
        los_users.append(user_coord)
    elif irs_los_flag:
        irs_los_users.append(user_coord)
    elif irs_nlos_flag:
        irs_nlos_users.append(user_coord)


# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Visual Studio code\\Test\\output_shapefile_chicago.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)
# Add random heights to the GeoDataFrame
heights_csv = "C:\\Visual Studio code\\Test\\ChicagoData.csv"
heights_df = pd.read_csv(heights_csv)

# Make sure that the CSV file has the same order of polygons as the shapefile
# If not, you might need to match them based on some identifier

# Assign heights from the CSV to the GeoDataFrame
obstacle_gdf['height'] = heights_df*0.3048

user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
user_heights = np.random.uniform(0,0.5, n_user)  # Random user heights
user_coords = list(zip(user_longitudes, user_latitudes, user_heights))

# Create empty lists for user markers
los_users = []          # LOS User from bs
irs_los_users = []      # IRS-LOS User
irs_nlos_users = []     # IRS-NLOS User
other_users = []        # Remaining Users

bs_coords = [(-87.36, 41.8835,150), (-87.826, 41.881,150), (-87.636, 41.884,150)]

irs_coords = [(-87.623, 41.884,100), (-87.627, 41.885,100), (-87.633, 41.881,100)]

bs_coordinates = [[(-87.36, 41.8835,150)], [(-87.36, 41.8835,150), (-87.826, 41.881,150)], [(-87.36, 41.8835,150), (-87.826, 41.881,150), (-87.636, 41.884,150)]]
irs_coordinates = [[(-87.623, 41.884,100)], [(-87.623, 41.884,100), (-87.627, 41.885,100)], [(-87.623, 41.884,100), (-87.627, 41.885,100), (-87.633, 41.881,100)]]
    
# Import necessary libraries

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

coverage_prob_np_1 = np.zeros((len(bs_coordinates), len(irs_coordinates)), dtype=object)

def process_coverage(bs_idx, bs_coords, irs_idx, irs_coords, n_iterations):
    N1 = N_val[0]
    n_bs = len(bs_coords)
    n_irs = len(irs_coords)
    rate = np.zeros(n_user)
    print(n_bs)
    final_rate = []
    coverage_prob = np.empty((n_user), dtype=object)
    
    for _ in range(n_iterations):
        user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
        user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
        user_heights = np.random.uniform(0,0.5, n_user)  # Random user heights
        user_coords = list(zip(user_longitudes, user_latitudes, user_heights))

        # Arrays to store LOS and NLOS users for each BS
        los_users_per_bs = [[] for _ in range(n_bs)]
        nlos_users_per_bs = [[] for _ in range(n_bs)]

        # Iterate through each user and check LOS from each BS
        for user_coord in user_coords:
            user_point = Point(user_coord[:2])
            for bs_index, bs_coord in enumerate(bs_coords):
                has_los = has_los_with_obstacles((bs_coord[0],bs_coord[1],bs_coord[2]), (user_point.x, user_point.y, user_coord[2]), obstacle_gdf['geometry'])
                if has_los:
                    los_users_per_bs[bs_index].append(user_coord)
                else:
                    nlos_users_per_bs[bs_index].append(user_coord)

        #//////////////////////////////////////////////////////////////////////////////////////////
        # Initialize sets to store common LOS users
        common_los_users = set(los_users_per_bs[0])
        common_nlos_users = set(nlos_users_per_bs[0])

            # Find common LOS users among all base stations
        for bs_index in range(1, n_bs):
            common_los_users &= set(los_users_per_bs[bs_index])
            common_nlos_users &= set(nlos_users_per_bs[bs_index])

        # Store remaining uncommon LOS users for each base station
        uncommon_los_users_per_bs1 = [list(set(los_users_per_bs[bs_index]) - common_los_users) for bs_index in range(n_bs)]

        # Convert the sets back to lists
        common_los_users_list = list(common_los_users)
        common_nlos_users_list = list(common_nlos_users)

        # Initialize lists to store users based on which BS offers less path loss
        los_users_per_bs = [[] for _ in range(n_bs)]
        dist = []
        # Iterate through common LOS users
        for user_coord in common_los_users_list:
            dist = haversine_distance(user_coord[1], user_coord[0], bs_coord[1], bs_coord[0])
            dist = np.sqrt(np.array(dist) ** 2 + (user_coord[2]-bs_coord[2]) ** 2)
            path_losses = [compute_pathloss(dist, a_L) for bs_coord in bs_coords]
            min_path_loss_bs_index = path_losses.index(min(path_losses))
            los_users_per_bs[min_path_loss_bs_index].append(user_coord)

        # Find the maximum length among all lists
        max_length = max(len(lst) for lst in uncommon_los_users_per_bs1)
        # Fill each list with zeros to make them equal in length
        equal_lists = [lst + [0] * (max_length - len(lst)) for lst in uncommon_los_users_per_bs1]
        # Convert the list of lists to a 2D array
        equal_array = [list(row) for row in equal_lists]
        equal_array = np.array(equal_array, dtype=object)
        # Convert single values of 0 to pairs with (0, 0) coordinates
        channels = np.array([[coord if isinstance(coord, tuple) else (0, 0) for coord in row] for row in equal_array],dtype = object)
        n_bs = len(bs_coords)
        num_channels = len(channels)
        common_location_sets = []

        # Compare and find common locations between each pair of channels
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                common_locations = find_common_locations(channels[i], channels[j])
                common_location_sets.append((i, j, common_locations))

        # Identify all common locations across all channel pairs
        all_common_locations = set()
        for i, j, common_locations in common_location_sets:
            all_common_locations.update(common_locations)

        # Assign common locations to the nearest base station for each channel
        common_location_by_base_station = {i: [] for i in range(num_channels)}
        min_distances = []
        for loc in all_common_locations:
            for bs_coord in bs_coords:
                min_distance = haversine_distance(loc[1], loc[0], bs_coord[1], bs_coord[0])
                min_distances.append(np.sqrt(np.array(min_distance) ** 2 + (user_coord[2]-bs_coord[2]) ** 2))

            closest_bs_index = np.argmin(min_distances)
            for i, j, common_locations in common_location_sets:
                if loc in common_locations:
                    common_location_by_base_station[i if i == closest_bs_index else j].append(loc)
        # Create a new list to hold modified channels with common locations removed
        modified_channels = []
        for i, channel in enumerate(channels):
            modified_channel = replace_common_locations(channel, all_common_locations)
            modified_channels.append(modified_channel)

        # Create a new list to hold channels with common locations removed
        channels_without_common = []
        for channel in modified_channels:
            channel_without_common = [loc for loc in channel if not np.array_equal(loc, [0, 0])]
            channels_without_common.append(channel_without_common)
        # Append the assigned common locations to respective base station channels in channels_without_common
        for i, common_locs in common_location_by_base_station.items():
            channels_without_common[i].extend(common_locs)

        # Initialize a list to store combined LOS users for each BS
        combined_los_users_per_bs = [[] for _ in range(n_bs)]

        # Combine common LOS users with uncommon LOS users
        for bs_index in range(n_bs):
            combined_los_users_per_bs[bs_index] = los_users_per_bs[bs_index] + channels_without_common[bs_index]

        #//////////////////////////////////////////////////////////////////////////////////////////

        # Initialize a list to store path losses for combined LOS users from each BS
        path_losses_per_bs = [[] for _ in range(n_bs)]

        # Iterate through combined LOS users for each BS
        for bs_index, combined_los_users in enumerate(combined_los_users_per_bs):
            for user_coord in combined_los_users:
                bs_coord = bs_coords[bs_index]
                bs_distance = haversine_distance(user_coord[1], user_coord[0], bs_coord[1], bs_coord[0])
                bs_distance = np.sqrt((bs_distance)**2 + (user_coord[2]-bs_coord[2]) ** 2)
                bs_path_loss = compute_pathloss(bs_distance, a_L)
                path_losses_per_bs[bs_index].append(bs_path_loss)

        # Initialize lists to store users based on LOS link with each IRS
        los_users_per_irs = [[] for _ in range(len(irs_coords))]
        nlos_users_per_irs = [[] for _ in range(len(irs_coords))]

        # Iterate through common NLOS users and check LOS with each IRS
        for user_coord in common_nlos_users_list:
            user_point = Point(user_coord[:2])
            for irs_index, irs in enumerate(irs_coords):
                has_los_irs = has_los_with_obstacles((irs[0], irs[1], irs[2]), (user_point.x, user_point.y, user_coord[2]), obstacle_gdf['geometry'])
                if has_los_irs:
                    los_users_per_irs[irs_index].append(user_coord)
                else:
                    nlos_users_per_irs[irs_index].append(user_coord)

        # Initialize sets to store common LOS users
        common_los_users = set(los_users_per_irs[0])
        common_nlos_users = set(nlos_users_per_irs[0])

        # Find common LOS users among all base stations
        for bs_index in range(1, n_irs):
            common_los_users &= set(los_users_per_irs[bs_index])
            common_nlos_users &= set(nlos_users_per_irs[bs_index])

        # Store remaining uncommon LOS users for each base station
        uncommon_los_users_per_irs1 = [list(set(los_users_per_irs[bs_index]) - common_los_users) for bs_index in range(n_irs)]

        # Convert the sets back to lists
        common_los_users_list = list(common_los_users)
        common_nlos_users_list = list(common_nlos_users)

        # Find the maximum length among all lists
        max_length1 = max(len(lst) for lst in uncommon_los_users_per_irs1)
        # Fill each list with zeros to make them equal in length
        equal_lists1 = [lst + [0] * (max_length1 - len(lst)) for lst in uncommon_los_users_per_irs1]
        # Convert the list of lists to a 2D array
        equal_array1 = [list(row) for row in equal_lists1]
        equal_array1 = np.array(equal_array1, dtype=object)
        # Convert single values of 0 to pairs with (0, 0) coordinates
        channels1 = np.array([[coord if isinstance(coord, tuple) else (0, 0) for coord in row] for row in equal_array1],dtype = object)
        n_irs = len(irs_coords)
        num_channels = len(channels1)
        common_location_sets = []
        # Compare and find common locations between each pair of channels
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                common_locations = find_common_locations(channels1[i], channels1[j])
                common_location_sets.append((i, j, common_locations))
        # Identify all common locations across all channel pairs
        all_common_locations = set()
        for i, j, common_locations in common_location_sets:
            all_common_locations.update(common_locations)

        # Assign common locations to the nearest base station for each channel
        common_location_by_base_station = {i: [] for i in range(num_channels)}
        min_distances = []
        for loc in all_common_locations:
            for irs_coord in irs_coords:
                min_distance = haversine_distance(loc[1], loc[0], irs_coord[1], irs_coord[0])
                min_distances.append(np.sqrt(np.array(min_distance) ** 2 + (user_coord[2]-irs_coord[2]) ** 2))

            closest_bs_index = np.argmin(min_distances)
            for i, j, common_locations in common_location_sets:
                if loc in common_locations:
                    common_location_by_base_station[i if i == closest_bs_index else j].append(loc)
        # Create a new list to hold modified channels with common locations removed
        modified_channels = []
        for i, channel in enumerate(channels1):
            modified_channel = replace_common_locations(channel, all_common_locations)
            modified_channels.append(modified_channel)

        # Create a new list to hold channels with common locations removed
        uncommon_los_users_per_irs = []
        for channel in modified_channels:
            channel_without_common = [loc for loc in channel if not np.array_equal(loc, [0, 0])]
            uncommon_los_users_per_irs.append(channel_without_common)
        # Append the assigned common locations to respective base station channels in channels_without_common
        for i, common_locs in common_location_by_base_station.items():
            uncommon_los_users_per_irs[i].extend(common_locs)

        # Initialize a list to store path losses for common IRS NLOS users from their nearest BS
        path_loss_common_irs_nlos_users_per_bs = [[] for _ in range(n_bs)]

        # Iterate through common IRS NLOS users
        for user_coord in common_nlos_users_list:
            nearest_bs_index = None
            min_bs_distance = float('inf')

            # Find the nearest base station for the user
            for bs_index, bs_coord in enumerate(bs_coords):
                bs_distance = haversine_distance(user_coord[1],user_coord[0], bs_coord[1], bs_coord[0])
                bs_distance = np.sqrt((bs_distance)**2 + (user_coord[2]-bs_coord[2]) ** 2)
                if bs_distance < min_bs_distance:
                    nearest_bs_index = bs_index
                    min_bs_distance = bs_distance

            # Calculate path loss from the nearest base station
            bs_path_loss = compute_pathloss(min_bs_distance, a_n)
            # Store path loss in the respective list for the base station
            path_loss_common_irs_nlos_users_per_bs[nearest_bs_index].append(bs_path_loss)

        # Initialize lists to store connections based on minimum path loss
        connections = []
        # Iterate through each IRS and its common LOS users
        for irs_index in range(len(irs_coords)):
            irs_los_users = common_los_users_list
            for user_coord in irs_los_users:
                for bs_index, bs_coord in enumerate(bs_coords):
                    bs_distance = haversine_distance(user_coord[1], user_coord[0], bs_coord[1], bs_coord[0])
                    bs_distance = np.sqrt((bs_distance)**2 + (user_coord[2]-bs_coord[2]) ** 2)


                    for other_irs_index in range(len(irs_coords)):
                        if len(irs_coords) != 1:
                            if irs_index == other_irs_index:
                                continue
                        other_irs_coord = irs_coords[other_irs_index]
                        bs_irs_distance = haversine_distance(other_irs_coord[1], other_irs_coord[0], bs_coord[1], bs_coord[0])
                      
                        bs_irs_distance = np.sqrt((bs_irs_distance)**2 + (other_irs_coord[2]-bs_coord[2]) ** 2)
                        #//////////////////////////////////////
                        irs_distance = haversine_distance(user_coord[1], user_coord[0], other_irs_coord[1], other_irs_coord[0])
                        irs_distance = np.sqrt((irs_distance)**2 + (other_irs_coord[2]-bs_coord[2]) ** 2)
                        
                        total_distance = bs_irs_distance* irs_distance
                        total_path_loss = compute_pathloss(total_distance, a_L)  + compute_pathloss(bs_distance, a_n)
                        connections.append((user_coord, bs_index, irs_index, total_path_loss))

        # Sort connections by total path loss and store the best combination for each user
        user_connections = {}  # Dictionary to store the best connection for each user
        for user_coord, bs_index, irs_index, total_path_loss in connections:
            if user_coord not in user_connections:
                user_connections[user_coord] = (bs_index, irs_index, total_path_loss)
            else:
                if total_path_loss < user_connections[user_coord][2]:
                    user_connections[user_coord] = (bs_index, irs_index, total_path_loss)

        best_connections_uncommon = {}
        # Iterate through each IRS and its uncommon LOS users
        for irs_index,irs_coord in enumerate(irs_coords):
            irs_uncommon_users = uncommon_los_users_per_irs[irs_index]
            for user_coord in irs_uncommon_users:
                best_bs_index = None
                min_path_loss = float('inf')
                for bs_index, bs_coord in enumerate(bs_coords):
                    bs_distance = haversine_distance(user_coord[1], user_coord[0], bs_coord[1], bs_coord[0])
                    bs_distance = np.sqrt((bs_distance)**2 + (user_coord[2]-bs_coord[2]) ** 2)
                    bs_irs_distance = haversine_distance(irs_coord[1], irs_coord[0], bs_coord[1], bs_coord[0])
                    bs_irs_distance = np.sqrt((bs_irs_distance)**2 + (irs_coord[2]-bs_coord[2]) ** 2)
                    irs_distance = haversine_distance(user_coord[1], user_coord[0], irs_coord[1], irs_coord[0])
                    irs_distance = np.sqrt((irs_distance)**2 + (irs_coord[2]-bs_coord[2]) ** 2)
                    
                    total_distance = bs_irs_distance * irs_distance
                    
                    bs_path_loss = compute_pathloss(bs_distance, a_n) + compute_pathloss(total_distance, a_L)
                    if bs_path_loss < min_path_loss:
                        min_path_loss = bs_path_loss
                        best_bs_index = bs_index
                    if best_bs_index is not None:
                        user_coord_tuple = tuple(user_coord)  # Convert NumPy array to tuple
                        best_connections_uncommon[user_coord_tuple] = (best_bs_index, irs_index, min_path_loss)

        # Initialize a list to store combined path losses for each BS
        combined_path_losses_per_bs = [[] for _ in range(n_bs)]

        # Combine path losses from users and common IRS NLOS users
        for bs_index in range(n_bs):
            combined_path_losses_per_bs[bs_index] = path_losses_per_bs[bs_index] + path_loss_common_irs_nlos_users_per_bs[bs_index]

        # Iterate through user_connections and store path losses in respective lists
        for user_coord, (bs_index, _, total_path_loss) in user_connections.items():
            combined_path_losses_per_bs[bs_index].append(total_path_loss)

        # Iterate through best_connections_uncommon and store path losses in respective lists
        for user_coord, (bs_index, _, min_path_loss) in best_connections_uncommon.items():
            combined_path_losses_per_bs[bs_index].append(min_path_loss)
    #------------------------------------------------------------------------------------------------------------------/////////////////
        if len(irs_los_users) == 0:
            N = 0
        else:
            N = int(N1*n_irs/len(irs_los_users))
    #------------------------------------------------------------------------------------------------------------------/////////////////
        # Rician Fading Channel for los user
        Pr_los = [[] for _ in range(n_bs)]
        for bs_index in range(n_bs):
            g_m = rician_fading_channel(len(combined_los_users_per_bs[bs_index]), K, omega)
            for pl_los in path_losses_per_bs[bs_index]:
                Pr_los[bs_index] = Pt_dB + 10*np.log10(g_m**2)- pl_los

        Pr_nlos = [[] for _ in range(n_bs)]
        for bs_index in range(n_bs):
            c_m = generate_rayleigh_fading_channel(len(path_loss_common_irs_nlos_users_per_bs[bs_index]), mu, std_dev)
            for pl_nlos in path_loss_common_irs_nlos_users_per_bs[bs_index]:
                Pr_nlos[bs_index] = Pt_dB + 10*np.log10(c_m**2)- pl_nlos

        best_connections_uncommon.update(user_connections)
        # Rayleigh Fading Channel for nlos user for direct path
        h_m = generate_rayleigh_fading_channel(len(best_connections_uncommon), mu, std_dev)
        h_m = h_m.reshape(1,-1)
        f_m = []
        fading = []
        for user in range(len(best_connections_uncommon)):
            fading.append(generate_nakagami_samples(m, omega, N))
        f_m = np.array(fading)
        f_m = f_m.reshape(N,len(best_connections_uncommon))

        f_m_transpose = np.transpose(f_m)
        # Generate the Nakagami Channel from base_station to the IRS
        g = generate_nakagami_samples(m, omega, N)
        g = g.reshape(N,1)

        r = []
        for row_index in range(len(best_connections_uncommon)):
            single_row = f_m_transpose[row_index,:]
            result = np.dot(single_row, g)
            r.append(result)
        r = np.squeeze(r)
        r = r.reshape(1,len(best_connections_uncommon))
        product = abs(r)

        PR_db1 = [[] for _ in range(n_bs)]
        i = 0
        # Iterate through user_connections and store path losses in respective lists
        for user_coord, (bs_index, irs_index, total_path_loss) in best_connections_uncommon.items():
            #print(user_coord)
            bs_coord = bs_coords[bs_index]
            bs_distance = haversine_distance(user_coord[1], user_coord[0], bs_coord[1],bs_coord[0])
            bs_distance = np.sqrt((bs_distance)**2 + (user_coord[2]-bs_coord[2]) ** 2)
            
            dt_pl1 = compute_pathloss(bs_distance, a_n)
            dt_pl = 10**(dt_pl1/10) # direct path loss
            id_pl = 10**((total_path_loss - dt_pl1)/10)            # indirect path loss
            PR_W = 10*(pt) * (h_m[0][i]**2)/(dt_pl) + pt*(product[0][i]**2)/(id_pl) # received power for LOS links
            #print('IRS loss', id_pl)
            #print('l2 ', dt_pl)
            PR_db1[bs_index].append(10*np.log10(PR_W))
            i = i + 1
            
        PR_final = [[] for _ in range(n_bs)]
        for bs_index in range(n_bs):
            PR_final[bs_index].extend(Pr_los[bs_index])
            PR_final[bs_index].extend(Pr_nlos[bs_index])
            PR_final[bs_index].extend(PR_db1[bs_index])

        #print('IRS',PR_db1)
            
            #print('shape of PR LOS ', np.shape(Pr_los[bs_index]),'shape of PR nlos ', np.shape(Pr_nlos[bs_index]),'shape of PR IRS ', np.shape(PR_db1[bs_index]))

        #Calculate the noise value
        noise = -174+10*np.log10(B) + NF
        SNR_matrix = [[] for _ in range(n_bs)]
        # Calculate SNR
        for bs_index in range(n_bs):
            SNR_matrix[bs_index] = compute_SNR(PR_final[bs_index], noise)
            

        # Calculate rate
        rate_matrix = [[] for _ in range(n_bs)]
        for bs_index in range(n_bs):
            rate_matrix[bs_index] = compute_rate(SNR_matrix[bs_index])

        flat_rate_final_1 = []
        flat_rate_final = []
        
        for sublist in rate_matrix:
            flat_rate_final.extend(sublist)
        
        np.random.shuffle(flat_rate_final)
        flat_rate_final_1 = flat_rate_final
        while len(flat_rate_final) < n_user:   
            print('F') 
            y = np.array(flat_rate_final_1[:n_user - len(flat_rate_final_1)])
            flat_rate_final = np.concatenate((flat_rate_final,y))
            flat_rate_final = flat_rate_final[:n_user]
            flat_rate_final = flat_rate_final.reshape(-1)           
            #print('final ',np.shape(flat_rate_final))
        final_rate.append(np.array(flat_rate_final))
    
        #print('foo ', np.shape(final_rate))

    rate = np.mean(final_rate,axis=0)
    rate = rate.reshape(-1)
    #print('Second rate ', np.shape(rate))
    
    coverage_prob = np.array(rate)
    return coverage_prob
    

# Create a pool of processes

  # You can adjust the number of processes
rate = []
bs_coordinates = [[(-87.36, 41.8835,150)], [(-87.36, 41.8835,150), (-87.826, 41.881,150)], [(-87.36, 41.8835,150), (-87.826, 41.881,150), (-87.636, 41.884,150)]]
irs_coordinates = [[(-87.623, 41.884,100)], [(-87.623, 41.884,100), (-87.627, 41.885,100)], [(-87.623, 41.884,100), (-87.627, 41.885,100), (-87.633, 41.881,100)]]

n_iterations = 5

for bs_idx, bs_coords in enumerate(bs_coordinates):
    n_bs = len(bs_coords)
    for irs_idx, irs_coords in enumerate(irs_coordinates):

        p1 = process_coverage(bs_idx, bs_coords, irs_idx, irs_coords, n_iterations)

        coverage_prob_np_1[bs_idx,irs_idx] = p1
print("Shape-1: ", np.shape(coverage_prob_np_1))
print("q is ", coverage_prob_np_1)
np.save('c2_rate_4.npy',coverage_prob_np_1)