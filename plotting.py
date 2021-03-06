import numpy as np
import matplotlib.pyplot as plt


def normalize_direction(u, v):
    normalized_u = u / (np.sqrt(u**2 + v**2) + .00000001)
    normalized_v = v / (np.sqrt(u**2 + v**2) + .00000001)
    return normalized_u, normalized_v


def plot_swarm(x, y, u, v, field_size, step, repulsion_radius,
               orientation_radius, attraction_radius, plot_radius):
    normalized_u, normalized_v = normalize_direction(u, v)
    plt.figure(1)
    plt.clf()
    plt.quiver(x, y, normalized_u, normalized_v)
    plt.scatter(x, y, color='b', s=20, marker="o")
    # plt.axis([-10, field_size+10, -10, field_size+10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Step: {}".format(step))
    if plot_radius:
        for i in range(len(x)):
            circle_r = plt.Circle((x[i], y[i]),
                                  radius=repulsion_radius,
                                  color='r',
                                  fill=False)
            circle_o = plt.Circle((x[i], y[i]),
                                  radius=repulsion_radius + orientation_radius,
                                  color='y',
                                  fill=False)
            circle_a = plt.Circle((x[i], y[i]),
                                  radius=repulsion_radius + orientation_radius
                                  + attraction_radius,
                                  color='g',
                                  fill=False)
            plt.gcf().gca().add_artist(circle_r)
            plt.gcf().gca().add_artist(circle_o)
            plt.gcf().gca().add_artist(circle_a)


def plot_eigenvalues(lall_eigen):
    plt.figure(2)
    plt.clf()
    #r, = plt.plot(range(len(replusion_eigen)), replusion_eigen, color="r")
    #o, = plt.plot(range(len(orientation_eigen)), orientation_eigen, color="y")
    #a, = plt.plot(range(len(attraction_eigen)), attraction_eigen, color="g")
    lall, = plt.plot(range(len(lall_eigen)), lall_eigen, color="g")
    plt.title("Fiedler Eigenvalues")
    #plt.legend([r, o, a], ['Repulsion', 'Orientation', 'Attraction'])


def plot(x,
         y,
         u,
         v,
         field_size,
         step,
         repulsion_radius,
         orientation_radius,
         attraction_radius,
         lall_eigen,
         plot_radius=False):
    plot_swarm(x, y, u, v, field_size, step, repulsion_radius,
               orientation_radius, attraction_radius, plot_radius)
    #plot_eigenvalues(lall_eigen)
    plt.show(block=False)
    plt.pause(.0002)



def plot_runs_eigenvalues(lall_eigen_runs):
  plt.figure(3)
  plt.clf()
  
  xs = range(lall_eigen_runs.shape[1])

  eigen_means = np.mean(lall_eigen_runs, axis=0)
  plt.plot(xs, eigen_means, label=f"average of {lall_eigen_runs.shape[0]} runs", color="g")

  std = np.std(lall_eigen_runs, axis=0)
  plt.fill_between(xs, eigen_means - 2 * std, eigen_means + 2 * std, label="2 standard deviations", color="b", alpha=.2)

  plt.legend()
  plt.title("Fiedler Eigenvalues")

  plt.show()
