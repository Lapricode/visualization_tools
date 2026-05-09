import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import tkinter.colorchooser as cc


def create_subplots(positions = [[1, 1, 1]], dimensions = [2], fig_width = 10, fig_height = 5, fig_dpi = 100):
    fig = plt.figure(figsize = (fig_width, fig_height), dpi = fig_dpi)
    subplots = []
    for k in range(len(positions)):
        if dimensions[k] == 2:
            subplots.append(fig.add_subplot(positions[k][0], positions[k][1], positions[k][2]))
        elif dimensions[k] == 3:
            subplots.append(fig.add_subplot(positions[k][0], positions[k][1], positions[k][2], projection = "3d"))
        else:
            print("Invalid dimension specified. Please specify 2 or 3.")
            subplots.append(None)
    return fig, subplots

def scatter_2d(frame, x, y, x_lim = None, y_lim = None, x_lim_offset = 0., y_lim_offset = 0., \
                x_ticks = None, y_ticks = None, equal_aspect = False, \
                plot_color = "blue", plot_style = "o,4,-,2", plot_label = "2d plot", plot_label_style = "Arial,9,bold", \
                title = "2d plot", x_label = "x", y_label = "y", title_style = "Arial,14,bold", axis_style = "Arial,10,normal", show_grid = True):
    plot_style = plot_style.split(",")
    points_marker, points_size, line_type, line_size = plot_style[0], int(plot_style[1]), plot_style[2], int(plot_style[3])
    title_style = title_style.split(",")
    title_font, title_font_size, title_font_weight = title_style[0], int(title_style[1]), title_style[2]
    axis_style = axis_style.split(",")
    axis_font, axis_font_size, axis_font_weight = axis_style[0], int(axis_style[1]), axis_style[2]
    plot_label_style = plot_label_style.split(",")
    plot_label_font, plot_label_font_size, plot_label_font_weight = plot_label_style[0], int(plot_label_style[1]), plot_label_style[2]
    frame.plot(x, y, color = plot_color, marker = points_marker, markersize = points_size, linestyle = line_type, linewidth = line_size, label = plot_label)
    frame.set_title(title, font = title_font, fontsize = title_font_size, fontweight = title_font_weight)
    frame.set_xlabel(x_label, font = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_ylabel(y_label, font = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    if x_lim is None: x_lim = [min(x), max(x), 0.]
    if y_lim is None: y_lim = [min(y), max(y), 0.]
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    margin = 1e-5
    frame.set_xlim([x_lim[0] - x_lim_offset / 2. * x_range - margin / 2., x_lim[1] + x_lim_offset / 2. * x_range + margin / 2.])
    frame.set_ylim([y_lim[0] - y_lim_offset / 2. * y_range - margin / 2., y_lim[1] + y_lim_offset / 2. * y_range + margin / 2.])
    if x_ticks is not None: frame.set_xticks(x_ticks)
    if y_ticks is not None: frame.set_yticks(y_ticks)
    frame.set_aspect("equal") if equal_aspect else frame.set_aspect("auto")
    frame.grid(show_grid)
    legend = frame.legend(loc = "best", shadow = True)
    for text in legend.get_texts():
        text.set_fontname(plot_label_font)
        text.set_fontsize(plot_label_font_size)
        text.set_fontweight(plot_label_font_weight)

def scatter_3d(frame, x, y, z, x_lim = None, y_lim = None, z_lim = None, x_lim_offset = 0., y_lim_offset = 0., z_lim_offset = 0., \
                x_ticks = None, y_ticks = None, z_ticks = None, equal_aspect = False, \
                plot_color = "blue", plot_style = "o,4,-,2", plot_label = "3d plot", plot_label_style = "Arial,9,bold", \
                title = "3d plot", x_label = "x", y_label = "y", z_label = "z", title_style = "Arial,14,bold", axis_style = "Arial,10,normal", show_grid = True):
    plot_style = plot_style.split(",")
    points_marker, points_size, line_type, line_size = plot_style[0], int(plot_style[1]), plot_style[2], int(plot_style[3])
    title_style = title_style.split(",")
    title_font, title_font_size, title_font_weight = title_style[0], int(title_style[1]), title_style[2]
    axis_style = axis_style.split(",")
    axis_font, axis_font_size, axis_font_weight = axis_style[0], int(axis_style[1]), axis_style[2]
    plot_label_style = plot_label_style.split(",")
    plot_label_font, plot_label_font_size, plot_label_font_weight = plot_label_style[0], int(plot_label_style[1]), plot_label_style[2]
    frame.plot(x, y, z, color = plot_color, marker = points_marker, markersize = points_size, linestyle = line_type, linewidth = line_size, label = plot_label)
    frame.set_title(title, fontname = title_font, fontsize = title_font_size, fontweight = title_font_weight)
    frame.set_xlabel(x_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_ylabel(y_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_zlabel(z_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    if x_lim is None: x_lim = [min(x), max(x)]
    if y_lim is None: y_lim = [min(y), max(y)]
    if z_lim is None: z_lim = [min(z), max(z)]
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]
    margin = 1e-5
    frame.set_xlim([x_lim[0] - x_lim_offset / 2. * x_range - margin / 2., x_lim[1] + x_lim_offset / 2. * x_range + - margin / 2.])
    frame.set_ylim([y_lim[0] - y_lim_offset / 2. * y_range - margin / 2., y_lim[1] + y_lim_offset / 2. * y_range + - margin / 2.])
    frame.set_zlim([z_lim[0] - z_lim_offset / 2. * z_range - margin / 2., z_lim[1] + z_lim_offset / 2. * z_range + - margin / 2.])
    if x_ticks is not None: frame.set_xticks(x_ticks)
    if y_ticks is not None: frame.set_yticks(y_ticks)
    if z_ticks is not None: frame.set_zticks(z_ticks)
    frame.set_aspect("equal") if equal_aspect else frame.set_aspect("auto")
    frame.grid(show_grid)
    legend = frame.legend(loc = "best", shadow = True)
    for text in legend.get_texts():
        text.set_fontname(plot_label_font)
        text.set_fontsize(plot_label_font_size)
        text.set_fontweight(plot_label_font_weight)

def surface_3d(frame, x, y, z_func, x_lim = None, y_lim = None, z_lim = None, x_lim_offset = 0., y_lim_offset = 0., z_lim_offset = 0., \
                x_ticks = None, y_ticks = None, z_ticks = None, equal_aspect = False, colormap = "viridis", fill_alpha = 0.7, edge_color = "none", edge_width = 0.5, \
                plot_label = "3d plot", plot_label_style = "Arial,9,bold", colorbar_label = "z", colorbar_label_style = "Arial,9,bold", \
                title = "3d plot", x_label = "x", y_label = "y", z_label = "z", title_style = "Arial,14,bold", axis_style = "Arial,10,normal", show_grid = True, \
                contour_levels = None, contour_color = "black", contour_width = 1.0):
    title_style = title_style.split(",")
    title_font, title_font_size, title_font_weight = title_style[0], int(title_style[1]), title_style[2]
    axis_style = axis_style.split(",")
    axis_font, axis_font_size, axis_font_weight = axis_style[0], int(axis_style[1]), axis_style[2]
    plot_label_style = plot_label_style.split(",")
    plot_label_font, plot_label_font_size, plot_label_font_weight = plot_label_style[0], int(plot_label_style[1]), plot_label_style[2]
    colorbar_label_style = colorbar_label_style.split(",")
    colorbar_label_font, colorbar_label_font_size, colorbar_label_font_weight = colorbar_label_style[0], int(colorbar_label_style[1]), colorbar_label_style[2]
    if x_lim is None: x_lim = [min(x), max(x)]
    if y_lim is None: y_lim = [min(y), max(y)]
    if z_lim is None: z_lim = [np.min(z_func(x, y)), np.max(z_func(x, y))]
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]
    margin = 1e-5
    frame.set_xlim([x_lim[0] - x_lim_offset / 2. * x_range - margin / 2., x_lim[1] + x_lim_offset / 2. * x_range + margin / 2.])
    frame.set_ylim([y_lim[0] - y_lim_offset / 2. * y_range - margin / 2., y_lim[1] + y_lim_offset / 2. * y_range + margin / 2.])
    frame.set_zlim([z_lim[0] - z_lim_offset / 2. * z_range - margin / 2., z_lim[1] + z_lim_offset / 2. * z_range + margin / 2.])
    if x_ticks is not None: frame.set_xticks(x_ticks)
    if y_ticks is not None: frame.set_yticks(y_ticks)
    if z_ticks is not None: frame.set_zticks(z_ticks)
    frame.set_aspect("equal") if equal_aspect else frame.set_aspect("auto")
    X, Y = np.meshgrid(x, y)
    Z = z_func(X, Y) if callable(z_func) else z_func
    surface = frame.plot_surface(X, Y, Z, cmap = colormap, alpha = fill_alpha, edgecolor = edge_color, linewidth = edge_width, antialiased = True)
    proxy_surface = plt.plot([], [], color = "black", marker = "none", linestyle = "none", label = plot_label)[0]
    if contour_levels is not None:
        frame.contour(X, Y, Z, levels = contour_levels, colors = contour_color, linewidths = contour_width)
        proxy_contour = plt.plot([], [], color = contour_color, marker = "none", linestyle = "solid", linewidth = contour_width, label = "Contour levels")[0]
    colorbar = plt.colorbar(surface, ax = frame)
    colorbar.set_label(colorbar_label, fontname = colorbar_label_font, fontsize = colorbar_label_font_size, fontweight = colorbar_label_font_weight)
    frame.set_title(title, fontname = title_font, fontsize = title_font_size, fontweight = title_font_weight)
    frame.set_xlabel(x_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_ylabel(y_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_zlabel(z_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.grid(show_grid)
    if contour_levels is not None: legend = frame.legend(loc = "best", shadow = True, handles = [proxy_surface, proxy_contour])
    else: legend = frame.legend(loc = "best", shadow = True, handles = [proxy_surface])
    for text in legend.get_texts():
        text.set_fontname(plot_label_font)
        text.set_fontsize(plot_label_font_size)
        text.set_fontweight(plot_label_font_weight)

def contour_2d(frame, x, y, z_func, x_lim = None, y_lim = None, x_lim_offset = 0., y_lim_offset = 0., levels = 25, \
                x_ticks = None, y_ticks = None, equal_aspect = False, colormap = "viridis", fill = False, fill_alpha = 0.7, \
                plot_label = "Contour plot", plot_label_style = "Arial,9,bold", colorbar_label = "z", colorbar_label_style = "Arial,9,bold", \
                title = "Contour plot", x_label = "x", y_label = "y", title_style = "Arial,14,bold", axis_style = "Arial,10,normal", show_grid = True):
    title_style = title_style.split(",")
    title_font, title_font_size, title_font_weight = title_style[0], int(title_style[1]), title_style[2]
    axis_style = axis_style.split(",")
    axis_font, axis_font_size, axis_font_weight = axis_style[0], int(axis_style[1]), axis_style[2]
    plot_label_style = plot_label_style.split(",")
    plot_label_font, plot_label_font_size, plot_label_font_weight = plot_label_style[0], int(plot_label_style[1]), plot_label_style[2]
    colorbar_label_style = colorbar_label_style.split(",")
    colorbar_label_font, colorbar_label_font_size, colorbar_label_font_weight = colorbar_label_style[0], int(colorbar_label_style[1]), colorbar_label_style[2]
    if x_lim is None: x_lim = [min(x), max(x)]
    if y_lim is None: y_lim = [min(y), max(y)]
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    margin = 1e-5
    frame.set_xlim([x_lim[0] - x_lim_offset / 2. * x_range - margin / 2., x_lim[1] + x_lim_offset / 2. * x_range + margin / 2.])
    frame.set_ylim([y_lim[0] - y_lim_offset / 2. * y_range - margin / 2., y_lim[1] + y_lim_offset / 2. * y_range + margin / 2.])
    if x_ticks is not None: frame.set_xticks(x_ticks)
    if y_ticks is not None: frame.set_yticks(y_ticks)
    frame.set_aspect("equal") if equal_aspect else frame.set_aspect("auto")
    X, Y = np.meshgrid(x, y)
    Z = z_func(X, Y) if callable(z_func) else z_func
    if fill:
        contours = frame.contourf(X, Y, Z, levels = np.linspace(np.min(Z), np.max(Z), levels), cmap = colormap, alpha = fill_alpha, linestyles = "solid", antialiased = True)
        colorbar = plt.colorbar(contours, ax = frame)
        colorbar.set_label(colorbar_label, fontname = colorbar_label_font, fontsize = colorbar_label_font_size, fontweight = colorbar_label_font_weight)
        # values_map = mcolors.Normalize(vmin = np.min(Z), vmax = np.max(Z))
        # colorbar = plt.colorbar(mappable = plt.cm.ScalarMappable(norm = values_map, cmap = plt.get_cmap(colormap)), ax = frame)
    else:
        contours = frame.contour(X, Y, Z, levels = np.linspace(np.min(Z), np.max(Z), levels), cmap = colormap, linestyles = "solid", linewidths = 1.5, antialiased = True)
        colorbar = plt.colorbar(contours, ax = frame)
        colorbar.set_label(colorbar_label, fontname = colorbar_label_font, fontsize = colorbar_label_font_size, fontweight = colorbar_label_font_weight)
    frame.set_title(title, fontname = title_font, fontsize = title_font_size, fontweight = title_font_weight)
    frame.set_xlabel(x_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_ylabel(y_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.grid(show_grid)
    proxy = plt.plot([], [], color = "black", marker = "none", linestyle = "none", label = plot_label)[0]
    legend = frame.legend(loc = "best", shadow = True, handles = [proxy])
    for text in legend.get_texts():
        text.set_fontname(plot_label_font)
        text.set_fontsize(plot_label_font_size)
        text.set_fontweight(plot_label_font_weight)

def vector_field(frame):
    pass

def bar_chart(frame, data, x_lim = None, y_lim = None, x_lim_offset = 0., y_lim_offset = 0., \
                x_ticks = None, y_ticks = None, counts_to_ticks = False, equal_aspect = False, bins = 10, density = False, orientation = "vertical", \
                plot_color = "blue", edge_color = "black", alpha = 0.7, plot_label = "3d plot", plot_label_style = "Arial,9,bold", \
                title = "Bar chart", x_label = "Values", y_label = "Frequency", title_style = "Arial,14,bold", axis_style = "Arial,10,normal", show_grid = True):
    title_style = title_style.split(",")
    title_font, title_font_size, title_font_weight = title_style[0], int(title_style[1]), title_style[2]
    axis_style = axis_style.split(",")
    axis_font, axis_font_size, axis_font_weight = axis_style[0], int(axis_style[1]), axis_style[2]
    plot_label_style = plot_label_style.split(",")
    plot_label_font, plot_label_font_size, plot_label_font_weight = plot_label_style[0], int(plot_label_style[1]), plot_label_style[2]
    counts, bins_edges = np.histogram(data, bins = bins, density = density)
    bin_width = bins_edges[1] - bins_edges[0]
    if orientation == "vertical":
        frame.hist(data, bins = bins, density = density, color = plot_color, edgecolor = edge_color, alpha = alpha, label = plot_label)
    elif orientation == "horizontal":    
        frame.barh(bins_edges[:-1] + bin_width / 2, counts, height = bin_width, color = plot_color, edgecolor = edge_color, alpha = alpha, label = plot_label)
    frame.set_title(title, fontname = title_font, fontsize = title_font_size, fontweight = title_font_weight)
    frame.set_xlabel(x_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    frame.set_ylabel(y_label, fontname = axis_font, fontsize = axis_font_size, fontweight = axis_font_weight)
    if x_lim is None: x_lim = [min(bins_edges), max(bins_edges)]
    if y_lim is None: y_lim = [0, max(counts)]
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    margin = 1e-5
    frame.set_xlim([x_lim[0] - x_lim_offset / 2. * x_range - margin / 2., x_lim[1] + x_lim_offset / 2. * x_range + margin / 2.])
    frame.set_ylim([y_lim[0] - y_lim_offset / 2. * y_range - margin / 2., y_lim[1] + y_lim_offset / 2. * y_range + margin / 2.])
    if x_ticks is None and counts_to_ticks: x_ticks = np.array(bins_edges[((counts > 0).nonzero()[0]).tolist() + [-1]]).reshape(-1)
    else: frame.set_xticks(x_ticks)
    if y_ticks is None and counts_to_ticks: y_ticks = np.array([0.] + counts.tolist()).reshape(-1)
    else: frame.set_yticks(y_ticks)
    frame.set_aspect("equal") if equal_aspect else frame.set_aspect("auto")
    frame.grid(show_grid)
    legend = frame.legend(loc = "best", shadow = True)
    for text in legend.get_texts():
        text.set_fontname(plot_label_font)
        text.set_fontsize(plot_label_font_size)
        text.set_fontweight(plot_label_font_weight)
    return counts, bins_edges

def pie_chart(frame, data, labels, colors = None, explode = None, start_angle = 0, pie_shadow = False, autopct = "%.1f%%",
                title = "Pie Chart", title_style = "Arial,14,bold",
                legend_style = "Arial,10,normal", show_legend = True):
    title_style = title_style.split(",")
    title_font, title_font_size, title_font_weight = title_style[0], int(title_style[1]), title_style[2]
    legend_style = legend_style.split(",")
    legend_font, legend_font_size, legend_font_weight = legend_style[0], int(legend_style[1]), legend_style[2]
    wedges, texts, autotexts = frame.pie(data, labels = labels, colors = colors, explode = explode, startangle = start_angle, shadow = pie_shadow, autopct = autopct)
    frame.set_title(title, fontname = title_font, fontsize = title_font_size, fontweight = title_font_weight)
    if show_legend:
        legend = frame.legend(loc = "best", fontsize = legend_font_size, title_fontsize = legend_font_size, title = "Categories", shadow = True)
        for text in legend.get_texts():
            text.set_fontname(legend_font)
            text.set_fontweight(legend_font_weight)
    for autotext in autotexts:
        autotext.set_fontsize(legend_font_size)
        autotext.set_fontname(legend_font)
    return wedges, texts, autotexts


# def plot_navigation_potential_field_gradients(frame):  # plot the gradient of the navigation function of the obstacles avoidance solver
#     points_max_norm = max([np.linalg.norm(self.hntf2d_solver.q_i[k]) for k in range(len(self.hntf2d_solver.q_i))] + [np.linalg.norm(self.hntf2d_solver.q_init), np.linalg.norm(self.hntf2d_solver.q_d)])  # calculate the maximum norm of the points
#     grid_points_size = self.navigation_field_plot_points_divs  # the number of points in the grid
#     grid_scale_factor = 1.5  # the scale factor of the grid
#     x = np.linspace(-grid_scale_factor * points_max_norm, grid_scale_factor * points_max_norm, grid_points_size)  # the x values
#     y = np.linspace(-grid_scale_factor * points_max_norm, grid_scale_factor * points_max_norm, grid_points_size)  # the y values
#     X, Y = np.meshgrid(x, y)  # the meshgrid of the x and y values
#     P = np.array([X.flatten(), Y.flatten()]).T  # the points to evaluate the navigation function
#     gradients = np.zeros((len(P), 2))  # the gradients of the navigation function
#     gradients_scaled = np.zeros((len(X.flatten()), 2))  # the scaled gradients of the navigation function
#     gradients_desired_norm = 1.2 * (2.0 * grid_scale_factor * points_max_norm) / grid_points_size  # the desired norm of the plotted gradients
#     reject_point = False  # the flag to reject the point
#     for k in range(len(P)):  # for each point p
#         for i in range(len(self.hntf2d_solver.q_i)):
#             if np.linalg.norm(P[k] - self.hntf2d_solver.q_i[i]) < 1.0 * (2.0 * grid_scale_factor * points_max_norm) / 100.0:  # if the point p is on an obstacle
#                 reject_point = True  # set the flag to reject the point
#                 break
#         if not reject_point:  # if the point p is not near an obstacle
#             gradients[k] = self.hntf2d_solver.navigation_psi_gradient(P[k])  # the gradient of the navigation function at point p
#             gradient_norm = np.sqrt(gradients[k, 0]**2 + gradients[k, 1]**2)  # the norm of the current gradient
#             gradients_scaled[k] = (gradients[k] / (gradient_norm + 1e-10)) * gradients_desired_norm  # the normalized and rescaled gradient of the navigation function at point p
#         reject_point = False  # reset the flag to reject the point
#     # plot the gradients
#     general_fontsize = 7; text_margin = 0.02
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     norms_cap_value = 100  # the cap value of the norms of the gradients
#     gradients_true_norms = np.linalg.norm(gradients, axis = 1)  # the true norms of the gradients
#     gradients_capped_norms = np.minimum(gradients_true_norms, norms_cap_value)  # the capped norms of the gradients
#     norms_map = mcolors.Normalize(vmin = 0.0, vmax = np.max(gradients_capped_norms))
#     color_map = plt.get_cmap("turbo")
#     ax.quiver(X, Y, -gradients_scaled[:, 0], -gradients_scaled[:, 1], color = color_map(norms_map(gradients_capped_norms)), angles = "xy", scale_units = "xy", scale = 1)
#     cbar = plt.colorbar(mappable = plt.cm.ScalarMappable(norm = norms_map, cmap = color_map), ax = ax)
#     cbar.set_label("True values of the gradients norms", fontsize = general_fontsize+2)
#     q_init_plot, = ax.plot([self.hntf2d_solver.q_init[0]], [self.hntf2d_solver.q_init[1]], "ms", markersize = 5)
#     q_d_plot, = ax.plot([self.hntf2d_solver.q_d[0]], [self.hntf2d_solver.q_d[1]], "gs", markersize = 5)
#     for i in range(len(self.hntf2d_solver.q_i)):
#         q_i_plot, = ax.plot([self.hntf2d_solver.q_i[i][0]], [self.hntf2d_solver.q_i[i][1]], "bo", markersize = 5)
#         ax.text(self.hntf2d_solver.q_i[i][0] + text_margin, self.hntf2d_solver.q_i[i][1] + text_margin, f"{i+1}", fontsize = general_fontsize)
#     if len(self.R2_plane_path_control_law_output) != 0:  # if a path (on the transformed R2 plane) has been calculated by the control law
#         R2planews_path_plot, = ax.plot(np.array(self.R2_plane_path_control_law_output)[:, 0], np.array(self.R2_plane_path_control_law_output)[:, 1], "black", linewidth = 2)
#         ax.legend([q_init_plot, q_d_plot, q_i_plot, R2planews_path_plot], ["Start", "Target", "Obstacles", "Path found"], fontsize = general_fontsize, loc = "upper right")
#     else:  # if no path (on the transformed R2 plane) has been calculated yet by the control law
#         ax.legend([q_init_plot, q_d_plot, q_i_plot], ["Start", "Target", "Obstacles"], fontsize = general_fontsize, loc = "upper right")
#     ax.set_title(f"Negative gradients of the navigation function\nk_d = {self.k_d:.2f}, k_i = {self.k_i}, w_phi = {self.w_phi:.2f}\n(press mouse right click on the plot to view the gradients)")
#     ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
#     fig.canvas.mpl_connect("button_press_event", lambda event: self.mark_navigation_field_gradients(event, ax))  # connect a function to the plot
