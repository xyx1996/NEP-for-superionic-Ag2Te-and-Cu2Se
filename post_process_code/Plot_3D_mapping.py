# ------------------------- #
# 2D view of 3D mapping     #
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from scipy.interpolate import griddata


def read_bins(folder):

    bin_edges = np.load(folder+"/bin_edges.npy", allow_pickle=True)

    return bin_edges

def read_data(folder,file,tag=None):
    eV_to_J = 1.602176634e-19  # eV to J conversion factor
    data = np.load(os.path.join(folder,file), allow_pickle=True)

    if tag is not None:
        data = data[tag]
        if 'enthalpy' in tag:
            data = data/eV_to_J

    return data.T


def slice_average_value(center_point, normal_vector, bins, bin_value, thickness):
    """
    计算切片内的平均应力值
    
    参数:
    center_point: 切片中心点 (cx, cy, cz)
    normal_vector: 切片法向向量 (nx, ny, nz)
    bins: 三维网格的边界点 (三个一维数组的元组)
    bin_value: 三维网格的应力值 (三维数组)
    thickness: 切片厚度
    
    返回:
    plane_value: 切片上的平均应力值网格
    points_in_slice: 切片内的点坐标
    """
    # 创建网格点坐标
    x_centers = (bins[0][1:] + bins[0][:-1]) / 2  # x方向中心点
    y_centers = (bins[1][1:] + bins[1][:-1]) / 2  # y方向中心点
    z_centers = (bins[2][1:] + bins[2][:-1]) / 2  # z方向中心点
    
    # 创建三维网格
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    grid_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    values = bin_value.flatten()

    # find the points that are close to the center point
    center_point = np.array(center_point, dtype=float)
    mask = np.all(np.isclose(grid_points, center_point), axis=1)
    closest_point = grid_points[mask]
    if len(closest_point) > 0:
        center_point_in_grid = closest_point[0]
    else:
        #print("Consider nearest neighbor search...")
        # Consider nearest neighbor search instead
        distances = np.linalg.norm(grid_points - center_point, axis=1)
        closest_idx = np.argmin(distances)
        center_point_in_grid = grid_points[closest_idx]
    #bin_resolution = np.array([bins[0][1] - bins[0][0], bins[1][1] - bins[1][0], bins[2][1] - bins[2][0]])
    #center_point_in_grid += 0.5 * (bin_resolution)
    #print("Center point change into:", center_point_in_grid)

    normal_vector = np.array(normal_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    vectors_to_points = grid_points - np.array(center_point_in_grid)
    distances = np.abs(np.dot(vectors_to_points, normal_vector))
    
    # 选择切片内的点 (距离 < 厚度/2)
    in_slice_mask = distances <= thickness / 2
    points_in_slice = grid_points[in_slice_mask]
    values_in_slice = values[in_slice_mask]
    
    # 创建切片投影平面
    # 创建平面坐标系 (找到两个正交于法向的向量)
    if abs(normal_vector[0]) < 0.1 and abs(normal_vector[1]) < 0.1:
        u_axis = np.array([1.0, 0.0, 0.0])  # 处理垂直法向
    else:
        u_axis = np.array([-normal_vector[1], normal_vector[0], 0.0])
        u_axis = u_axis / np.linalg.norm(u_axis)
    
    v_axis = np.cross(normal_vector, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)
    
    # 将点投影到切平面
    u_coords = np.dot(points_in_slice - center_point_in_grid, u_axis)
    v_coords = np.dot(points_in_slice - center_point_in_grid, v_axis)

    # 创建网格用于插值
    u_min, u_max = np.min(u_coords), np.max(u_coords)
    v_min, v_max = np.min(v_coords), np.max(v_coords)
    
    # 扩展10%边界
    u_range = u_max - u_min
    v_range = v_max - v_min
    u_min, u_max = u_min - 0.05*u_range, u_max + 0.05*u_range
    v_min, v_max = v_min - 0.05*v_range, v_max + 0.05*v_range
    
    # 创建二维网格
    grid_u, grid_v = np.mgrid[u_min:u_max:100j, v_min:v_max:100j]
    
    # 插值到二维网格
    plane_value = griddata(
        (u_coords, v_coords), values_in_slice, 
        (grid_u, grid_v), method='linear', fill_value=np.nan
    )

    return plane_value, (grid_u, grid_v), points_in_slice, center_point_in_grid


def plot_3D_slice(plane_value_plot, plane_grid, points_in_slice, center_xyz, normal_xyz, thickness, tag=None, save_tag=None):
    """
    绘制3D切片可视化
    
    参数:
    plane_value: 切片上的平均应力值网格 (2D数组)
    plane_grid: 二维网格坐标 (u, v) 每个都是2D数组
    points_in_slice: 切片内的点坐标 (N,3)
    center_xyz: 切片中心点 (x, y, z)
    normal_xyz: 切片法向向量 (x, y, z)
    thickness: 切片厚度
    tag: 数据标签 (可选)
    """
    
    n_subplots = len(plane_value_plot)+1
    size_width = n_subplots * 5

    grid_u, grid_v = plane_grid
    fig = plt.figure(figsize=(size_width, 4))

    # ================ Plot 3D slice ================
    ax_3d = fig.add_subplot(1, n_subplots, 1, projection='3d')
    ax_3d.set_title(f"3D Slice: Center=({center_xyz[0]:.2f},{center_xyz[1]:.2f},{center_xyz[2]:.2f})\nNormal={normal_xyz} Thickness={thickness}", fontsize=10)
    ax_3d.set_xlabel('X-axis')
    ax_3d.set_ylabel('Y-axis')
    ax_3d.set_zlabel('Z-axis')
    
    #----plot points in slice---
    all_points = []
    if len(points_in_slice) > 0:
        all_points.append(points_in_slice)
    
    normal_vector = np.array(normal_xyz, dtype=float)
    normal_vector /= np.linalg.norm(normal_vector)
    
    if abs(normal_vector[0]) < 1e-6 and abs(normal_vector[1]) < 1e-6:
        u_axis = np.array([1.0, 0.0, 0.0])
    else:
        u_axis = np.array([-normal_vector[1], normal_vector[0], 0.0])
        u_axis /= np.linalg.norm(u_axis)
    
    v_axis = np.cross(normal_vector, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    
    u_min, u_max = np.nanmin(grid_u), np.nanmax(grid_u)
    v_min, v_max = np.nanmin(grid_v), np.nanmax(grid_v)
    uu, vv = np.meshgrid(np.linspace(u_min, u_max, 20), 
                         np.linspace(v_min, v_max, 20))
    
    xx = center_xyz[0] + u_axis[0]*uu + v_axis[0]*vv
    yy = center_xyz[1] + u_axis[1]*uu + v_axis[1]*vv
    zz = center_xyz[2] + u_axis[2]*uu + v_axis[2]*vv
    
    all_points.append(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
    
    if len(points_in_slice) > 0:
        ax_3d.scatter(
            points_in_slice[:, 0], points_in_slice[:, 1], points_in_slice[:, 2],
            c='blue', alpha=0.3, s=5, label='Points in Slice'
        )
    
    # plot plane and normal vector
    ax_3d.plot_surface(xx, yy, zz, alpha=0.5, color='green', label='Slice Plane')
    ax_3d.quiver(
        center_xyz[0], center_xyz[1], center_xyz[2],
        normal_vector[0]*10, normal_vector[1]*10, normal_vector[2]*10,
        color='red', lw=5, label='Normal'
    )
    ax_3d.legend(loc='upper left', fontsize=10)
    
    if all_points:

        all_points = np.vstack(all_points)
        
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        ax_3d.set_box_aspect([1, 1, 1])
    
    # ================ Plot 2D slices ================
    for i, plane_value in enumerate(plane_value_plot):
        ax_2d = fig.add_subplot(1, n_subplots, i+2)
        #ax_2d.set_title(f"Slice Projection Thickness={thickness}", fontsize=10)
        ax_2d.set_xlabel('U-axis')
        ax_2d.set_ylabel('V-axis')
        
        if not np.isnan(plane_value).all():
            # 提取唯一的U和V坐标
            # 假设网格是规则矩形网格，每行有相同的U坐标，每列有相同的V坐标
            u_coords = grid_u[:, 0]  # 第一行的U坐标
            v_coords = grid_v[0, :]  # 第一列的V坐标

            im = NonUniformImage(
                ax_2d, 
                interpolation='bilinear',
                cmap='viridis',
                extent=(u_coords.min(), u_coords.max(), v_coords.min(), v_coords.max())
            )
            
            im.set_data(u_coords, v_coords, plane_value)
            im.set_clim(vmin=np.nanmin(plane_value), vmax=np.nanmax(plane_value))
            ax_2d.add_image(im)
            ax_2d.set_xticks(np.arange(-10.0, 11.0, 5.0))
            ax_2d.set_yticks(np.arange(-10.0, 11.0, 5.0))

            ax_2d.set_aspect('equal')
            cbar = plt.colorbar(im, ax=ax_2d)
            cbar.set_label(tag[i] if tag[i] else 'Value', fontsize=10)

            '''
            # 添加等高线
            contour = ax_2d.contour(
                grid_u, grid_v, plane_value, 
                levels=8, colors='black', linewidths=0.8
            )
            ax_2d.clabel(contour, inline=True, fontsize=8)
            # 标记平均值位置
            avg_u = np.nanmean(grid_u)
            avg_v = np.nanmean(grid_v)
            avg_value = np.nanmean(plane_value)
            ax_2d.scatter(avg_u, avg_v, s=200, c='red', edgecolor='white', marker='*')
            ax_2d.text(
                avg_u, avg_v, f'Average: {avg_value:.2f}', 
                color='white', fontsize=12, ha='center', va='bottom',
                bbox=dict(facecolor='red', alpha=0.8)
            )
            '''
    

    plt.subplots_adjust(
        left=0.01,      # 左边距
        right=0.95,     # 右边距
        bottom=0.15,    # 底边距
        top=0.85,       # 顶边距
        wspace=0.2      # 子图水平间距
    )
    # get string from normal vector
    normal_str = "_".join([f"{int(n)}" for n in normal_xyz])
    if tag is not None:
        tag_str = "_".join(tag)
        figure_name = f"3D_slice_{save_tag}_{tag_str}_{normal_str}_{center_point_real}.png"
    else:
        figure_name = f"3D_slice_no_tag.png"
    plt.savefig(os.path.join(folder, figure_name), dpi=600)
    return

def integrate_volume_value(folder, bin_edges, data, center_point, tag, r_step=0.1, save_tag=None):
    """
    3D radial integration of volume values around a center point
    
    Args:
        bin_edges (list): List of three 1D arrays [x_edges, y_edges, z_edges]
        data (ndarray): 3D array of values to integrate
        center_point (array-like): [x, y, z] coordinates of center point
    
    Returns:
        r_unique (ndarray): Unique radial distances
        local_integrate (ndarray): Integrated values at each radial distance
    """
    # Convert to numpy arrays
    center_point = np.array(center_point)
    x_edges, y_edges, z_edges = bin_edges
    bin_resolution = np.array([x_edges[1] - x_edges[0], y_edges[1] - y_edges[0], z_edges[1] - z_edges[0]])
    center_point += 0.5 * (bin_resolution)

    # Calculate bin centers (where data values are defined)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Create grid of bin centers
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Calculate distances from center to all bin centers
    r_all = np.sqrt((X - center_point[0])**2 + 
                    (Y - center_point[1])**2 + 
                    (Z - center_point[2])**2)
    #print(X.shape,r_all.shape)
    # Calculate domain extents and set cutoff radius
    domain_extents = np.array([
        center_point[0] - x_edges[0],
        center_point[1] - y_edges[0],
        center_point[2] - z_edges[0],
        center_point[0] - x_edges[-1],
        center_point[1] - y_edges[-1],
        center_point[2] - z_edges[-1],
    ])
    domain_extents = np.abs(domain_extents)
    cut_global = np.min(domain_extents)
    
    # Create mask for points within cutoff
    mask = r_all <= cut_global
    r_masked = r_all[mask]
    data_masked = data[mask]

    r_renormalized = np.arange(0, np.floor(np.max(r_masked)), r_step)

    # Calculate integrated values at each unique radius
    local_integrate = np.zeros_like(r_renormalized)
    r_acumulated = np.zeros_like(r_renormalized)
    count_local = np.zeros_like(r_renormalized)

    for i, r in enumerate(r_renormalized):
        # Handle floating point precision with tolerance
        mask_local = np.isclose(r_masked, r, atol=r_step*0.5)
        count_local[i] = np.count_nonzero(mask_local)
        # dV = 4/3*np.pi*(r*r*r-(r_unique[i-1]**3)) if i > 0 else 4/3*np.pi*r*r*r
        local_integrate[i] = np.sum(data_masked[mask_local])
        if i == 0:
            r_acumulated[i] = local_integrate[i]
        else:
            r_acumulated[i] = r_acumulated[i-1] + local_integrate[i]

        local_integrate[i] = local_integrate[i] / count_local[i] if count_local[i] > 0 else 0.0

    #for i, r in enumerate(r_renormalized):
        #r_acumulated[i] = r_acumulated[i] / np.sum(count_local[:i+1]) if np.sum(count_local[:i+1]) > 0 else 0.0

    system_average = np.sum(data) / np.count_nonzero(data)
    mask_save = count_local > 0
    np.savetxt(os.path.join(folder, f"local_integrate_{save_tag}_{tag}.txt"),
               np.column_stack((r_renormalized[mask_save], local_integrate[mask_save], r_acumulated[mask_save], count_local[mask_save])),fmt='%10.6f')
    # Create figure and axis for better control
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot local values as circles
    ax.scatter(
        r_renormalized[mask_save], 
        local_integrate[mask_save],
        marker='o',
        facecolors='none',
        edgecolors='b',  # More explicit than color
        linewidths=0.5,  
        s=15,
        label='Local'
    )

    # Plot accumulated values as triangles
    ax.scatter(
        r_renormalized[mask_save], 
        r_acumulated[mask_save],
        marker='^',
        facecolors='none',
        edgecolors='r',  # More explicit than color
        linewidths=0.5,  
        s=15,
        label='Accumulated'
    )

    # Add system average reference line
    ax.axhline(
        y=system_average,
        color='k',
        linestyle='--',
        label='System Average'
    )

    # Set labels and title
    ax.set_xlabel('Radial Distance (Å)')
    ax.set_ylabel('Integrated Value')
    ax.set_title(f'Radial Integration of {tag} Density')  # Modern f-string formatting

    # Add legend and adjust layout
    ax.legend()
    plt.tight_layout()

    # Save figure with explicit bbox_inches
    plt.savefig(
        os.path.join(folder, f"radial_integration_{save_tag}_{tag}.png"),
        dpi=600,
        bbox_inches='tight'  # Ensure no elements are cut off
    )

    # Close figure to free memory
    plt.close(fig)

    return

def plot_excess_enthalpy(folder, system_enthalpy, type_enthalpy):

    #Plot excess enthalpy as a function of radial distance from a center point.
    data_system = np.loadtxt(os.path.join(folder, system_enthalpy))
    data_type = np.loadtxt(os.path.join(folder, type_enthalpy))

    x_axis = data_system[:, 0]
    h_rho_system = data_system[:, 1]
    h_rho_type = data_type[:, 1]
    excess_enthalpy_density = h_rho_type - h_rho_system
    h_acc_system = data_system[:, 2]
    h_acc_type = data_type[:, 2]
    excess_enthalpy_acc = h_acc_type - h_acc_system

    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    ax[0].scatter(x_axis, excess_enthalpy_density,
                    label='Excess enthalpy density',
                    marker='^',
                    facecolors='none',
                    edgecolors='r',  # More explicit than color
                    linewidths=0.5,  
                    s=15)
    ax[0].scatter(x_axis, h_rho_type,
                    label='Fluid enthalpy density',
                    marker='s',
                    facecolors='none',
                    edgecolors='b',  # More explicit than color
                    linewidths=0.5,  
                    s=15)
    ax[0].scatter(x_axis, h_rho_system,
                    label='System enthalpy density',
                    marker='o',
                    facecolors='none',
                    edgecolors='g',  # More explicit than color
                    linewidths=0.5,  
                    s=15)
    ax[0].set_xlabel('Radial distance (Å)')
    ax[0].set_ylabel('Local density')
    ax[0].legend()

    ax[1].scatter(x_axis, excess_enthalpy_acc,
                    label='Excess enthalpy',
                    marker='^',
                    facecolors='none',
                    edgecolors='b',  # More explicit than color
                    linewidths=0.5,  
                    s=15)
    ax[1].scatter(x_axis, h_acc_type,
                    label='Fluid enthalpy',
                    marker='s',
                    facecolors='none',
                    edgecolors='r',  # More explicit than color
                    linewidths=0.5,  
                    s=15)
    ax[1].scatter(x_axis, h_acc_system,
                    label='System enthalpy',
                    marker='o',
                    facecolors='none',
                    edgecolors='g',  # More explicit than color
                    linewidths=0.5,  
                    s=15)
    ax[1].set_xlabel('Radial distance (Å)')
    ax[1].set_ylabel('Enthalpy')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "excess_enthalpy.png"), dpi=600)
    plt.close(fig)
    print("Excess enthalpy plot saved.")

    return

if __name__ == "__main__":

    system = "result_time_total_ave.npz"
    type = "result_time_ave_Ag.npz" # should be the moving particle type

    tag = ["local_temperature", "local_enthalpy"]

    file = [system, type]

    # Assuming the data contains center point and normal vector
    center_point = [23, 26, 26]         # Example center point
    normal_vector = [1, 0, 0]           # Example normal vector
    thickness =  1                      # Example thickness
    r_step = 0.2

    head_folder = "NVT/md_batch_split"
    sub_list = ["0.6","0.7","0.8","0.9"]

    for sub in sub_list:
        folder = f"{head_folder}/{sub}"
        print(f"Processing subfolder: {sub} in {head_folder}")
        
        # Read bin edges and data
        bin_edges = read_bins(folder)
        data_system = read_data(folder, "result_time_total_ave.npz", tag="local_temperature")
        data_fluid = read_data(folder, "result_time_ave_Ag.npz", tag="local_enthalpy")

        for j in range(len(file)):
            if j == 0:
                save_tag = 'system'
            else:
                save_tag = 'fluid'
            print(f"Processing file: {file[j]} with save tag: {save_tag}")

            for i in range(len(tag)):
                data = read_data(folder, file[j], tag=tag[i])
                plane_value, plane_grid, points_in_slice, center_point_real = slice_average_value(center_point, normal_vector, bin_edges, data, thickness)
                integrate_volume_value(folder, bin_edges, data, center_point_real, tag[i], r_step=r_step, save_tag=save_tag)
                if i == 0:
                    plane_value_plot = np.zeros((len(tag), *plane_value.shape))
                plane_value_plot[i] = plane_value

            plot_3D_slice(plane_value_plot, plane_grid, points_in_slice, center_point_real, normal_vector, thickness, tag=tag, save_tag=save_tag)

        system_enthalpy = "local_integrate_system_local_enthalpy.txt"
        type_enthalpy = "local_integrate_fluid_local_enthalpy.txt"
        plot_excess_enthalpy(folder, system_enthalpy, type_enthalpy)