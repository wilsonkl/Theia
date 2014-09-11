.. highlight:: c++

.. default-domain:: cpp

.. _documentation-sfm:

===========================
Structure from Motion (SfM)
===========================

Theia has implementations of many common Structure from Motion (sfm) algorithms. We
attempt to use a generic interface whenever possible so as to maximize
compatibility with other libraries.

You can include the SfM module in your code with the following line:

.. code-block:: c++

  #include <theia/sfm.h>

Projection Matrix
=================

We provide two convenience matrices that are commonly used in multiview geometry. The first is a :class:`TransformationMatrix` which is an affine transformation matrix composed of rotation and translation of the form: :math:`\left[R | t\right]`, i.e., the extrinsic parameters of a camera. The :class:`TransformationMatrix` is merely a typedef of the Eigen affine transformation matrix. Similarly, a :class:`ProjectionMatrix` class is defined that representsthe intrinsic and extrinsic parameters, nameley matrices of the form: :math:`K\left[R | t \right]` where :math:`K` is a 3x3 matrix of the camera intrinsics (e.g., focal length, principle point, and radial distortion). The :class:`ProjectionMatrix` is merely a typedef of an Eigen 3x4 matrix.

Camera and CameraPose
=====================

At the core of SfM are cameras which provide us with observations of 3D points. Theia uses a :class:`Camera` struct to maintain all imaging and view information. This includes information about the image captured, the feature positions (both in the image planes and in 3D space), and feature descriptors.

.. code-block:: c++

  struct Camera {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Camera() {}
    explicit Camera(const CameraPose& pose) : pose_(pose) {}

    // The camera pose describing the position, orientation, and intrinsics of the
    // camera.
    CameraPose pose_;

    // Width of the camera in pixels.
    int width_;
    // Height of the camera in pixels.
    int height_;

    // Original measured feature positions.
    std::vector<Eigen::Vector2d> feature_position_2D_distorted_;
    // Feature positions after correcting for radial distortion.
    std::vector<Eigen::Vector2d> feature_position_2D_;

    // Descriptors.
    std::vector<Eigen::VectorXf> descriptors_;
    std::vector<Eigen::BinaryVectorX> binary_descriptors_;

    // 3D feature IDs. This assumes that a container of 3D feature positions is
    // owned elsewhere. This is essentially the track ID, and the main usage is
    // for reconstructions where many 2D points correspond to a single 3D point.
    std::vector<size_t> feature_3D_ids_;

    // If the 3D positions are stored locally, then we can keep them in a
    // container here.
    std::vector<Eigen::Vector3d> feature_positions_3D_;
  };

The pose of the camera is contained in the :class:`CameraPose` object. This
class contains extrinsic (rotation and translation) and intrinsic calibration
(focal length, principle point, radial distortion) information for the camera,
and provides many convenience functions for various image and point
transformations.

  .. class:: CameraPose

    .. cpp:function:: CameraPose()

      The default constructor. Sets all values to identity values.

    The :class:`CameraPose` class must be initialized with the
    :func:`InitializePose` method rather than with the constructor. There are
    several variations of this method so as to be flexible to the information
    that the user has available:

    .. cpp:function:: void InitializePose(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation, const Eigen::Matrix3d& calibration, const double k1, const double k2, const double k3, const double k4)

      Initialize the camera pose with the full extrinsic (rotation and
      translation) and intrinsic (calibration matrix and radial distortion)
      parameters. The extrinsic parameters should provide world-to-camera
      transformations.

    .. cpp:function:: void InitializePose(const Eigen::Matrix<double, 3, 4>& projection_matrix, const double k1, const double k2, const double k3, const double k4)

      Initialize the pose with a projection matrix given by P = K * [R | t],
      where K is the calibration matrix, R is the rotation matrix, and t is the
      translation. The projection matrix provided should be a world-to-image
      transformation (as opposed to world-to-camera).

    .. cpp:function:: void InitializePose(const Eigen::Matrix<double, 3, 4>& transformation_matrix, const Eigen::Matrix3d& calibration, const double k1, const double k2, const double k3, const double k4)

      Initialize the pose with a given transformation matrix that defines the
      world-to-camera transformation.

    .. cpp:function:: void InitializePose(const CameraPose& pose)

      Copy constructor.

    It is important to be able to access components of the camera pose, so we
    provide getter functions for all relative information:

    .. cpp:function:: Eigen::Matrix3d rotation_matrix() const

      Get the rotation componenet of the transformation matrix.

    .. cpp:function:: Eigen::Vector3d translation() const

      Get the translation componenet of the transformation matrix.

    .. cpp:function:: Eigen::Vector3d position() const

      Get the camera position in the world coordinate system defined as
      position = -R' * t.

    .. cpp:function:: Eigen::Matrix3d calibration_matrix() const

      Get the 3x3 camera calibration matrix defined by K = diag(f, f, 1).

    .. cpp:function:: double focal_length() const

      Get the focal length of the camera.

    .. cpp:function:: void radial_distortion(double* k1, double* k2, double* k3, double* k4) const

      Returns the radial distortion parameters.

    .. cpp:function:: Eigen::Matrix<double, 3, 4> projection_matrix() const

      Returns the full projection matrix that describes the camera pose.

    .. cpp:function:: Eigen::Matrix<double, 3, 4> transformation_matrix() const

      Returns the transformation matrix :math:`T = [R | t]` i.e., the extrinsic parameters.

    Finally, the most important functionality of a camera is that it projects
    points in the world into the image. Modeling this projection is a crucial
    part of structure from motion (and all projective geometry!), so we provide
    transformation functions to perform these tasks for you. We define three
    coordinate systems: the world coordinate system, camera coordinate system,
    and image coordinate system. The world coordinate system is defined as the
    3D coordinate system relative to some world origin. 3D points in SfM models
    are typically defined with respect to the world coordinate system. The
    camera coordinate system is the coordinate system centered around the
    camera. That is, with the camera at the origin looking down the
    z-axis. Finally, the image coordinate system is the camera coordinate system
    projected onto the image plane. The image coordinate system is defined in
    pixels, and may only be reconciled with respect to the other coordinate
    systems when the intrinsic paramters are known.

    So, given a point in the world coordinate system, :math:`X_w`, we can
    transform that point with a translation and rotation :math:`T=[R | t]` such
    that :math:`X_c = T * X_w` is a point in the camera coordinate system. To
    transform :math:`X_c` into image coordinates, we must apply the camera
    calibration matrix, :math:`K` such that :math:`X_i = K * T * X_w` is a point
    in the image plane (in pixels).

    The functions below provide these transformations for a single point, as
    well as optimized transformations for transforming multiple points at the
    same time.

    .. cpp:function:: void WorldToCamera(const Eigen::Vector3d& world_point, Eigen::Vector3d* camera_point) const

      Transforms a point from the world coordinate system to the camera
      coordinate system.

    .. cpp:function:: void WorldToCamera(const std::vector<Eigen::Vector3d>& world_point, std::vector<Eigen::Vector3d>* camera_point) const

      Transformation method for multiple points.

    .. cpp:function:: void CameraToWorld(const Eigen::Vector3d& camera_point, Eigen::Vector3d* world_point) const

      Transforms a point from the camera coordinate system to the world
      coordinate system.

    .. cpp:function:: void CameraToWorld(const std::vector<Eigen::Vector3d>& camera_point, std::vector<Eigen::Vector3d>* world_point) const

      Transformation method for multiple points.

    .. cpp:function:: void CameraToImage(const Eigen::Vector3d& camera_point, Eigen::Vector2d* image_point) const

      Projects the 3D points in camera coordinates into the image plane using the
      calibration matrix of the camera.

    .. cpp:function:: void CameraToImage(const std::vector<Eigen::Vector3d>& camera_point, std::vector<Eigen::Vector2d>* image_point) const

      Projection method for multiple points. NOTE: this method is void, and does
      not indicate whether points are in front of behind the camera.

    .. cpp:function:: bool WorldToImage(const Eigen::Vector3d& world_point, Eigen::Vector2d* image_point) const

      Projects the 3D points in world coordinates into the image plane using the
      projection matrix of the camera. Returns true if the point is in front of
      the camera and false otherwise.

    .. cpp:function:: void WorldToImage(const std::vector<Eigen::Vector3d>& world_point, std::vector<Eigen::Vector2d>* image_point) const

      Projection method for multiple points. NOTE: this method is void, and does
      not indicate whether points are in front of behind the camera.

    Correcting radial distortion can be a common operation for SfM so that the
    images may be as geomtetrically correct as possible. The following two
    functions will undistort image points based on the intrinsic paramters of
    the camera.

    .. cpp:function:: void UndistortImagePoint(const Eigen::Vector2d& distorted_point, Eigen::Vector2d* undistorted_point) const

      Undistorts the image point using the radial distortion parameters.

    .. cpp:function:: void UndistortImagePoint(const std::vector<Eigen::Vector2d>& distorted_point, std::vector<Eigen::Vector2d>* undistorted_point) const

      Undistort multiple points at the same time.

2-View Triangulation
====================

  Triangulation in structure from motion calculates the 3D position of an image
  coordinate that has been tracked through several, if not many, images.

  .. cpp:function:: bool Triangulate(const ProjectionMatrix& pose_left, const ProjectionMatrix& pose_right, const Eigen::Vector2d& point_left, const Eigen::Vector2d& point_right, Eigen::Vector3d* triangulated_point)

    2-view triangulation using the DLT method described in
    [HartleyZisserman]_. The poses are the (potentially calibrated) poses of the
    two cameras, and the points are the 2D image points of the matched features
    that will be used to triangulate the 3D point. If there was an error computing
    the triangulation (e.g., the point is found to be at infinity) then ``false``
    is returned. On successful triangulation, ``true`` is returned.

N-View Triangulation
====================

  .. cpp:function:: bool TriangulateNViewSVD(const std::vector<ProjectionMatrix>& poses, const std::vector<Eigen::Vector2d>& points, Eigen::Vector3d* triangulated_point)
  .. cpp:function:: bool TriangulateNView(const std::vector<ProjectionMatrix>& poses, const std::vector<Eigen::Vector2d>& points, Eigen::Vector3d* triangulated_point)

    We provide two N-view triangluation methods that minimizes an algebraic
    approximation of the geometric error. The first is the classic SVD method
    presented in [HartleyZisserman]_. The second is a custom algebraic
    minimization. Note that we can derive an algebraic constraint where we note
    that the unit ray of an image observation can be stretched by depth
    :math:`\alpha` to meet the world point :math:`X` for each of the :math:`n`
    observations:

    .. math:: \alpha_i \bar{x_i} = P_i X,

    for images :math:`i=1,\ldots,n`. This equation can be effectively rewritten as:

    .. math:: \alpha_i = \bar{x_i}^\top P_i X,

    which can be substituted into our original constraint such that:

    .. math:: \bar{x_i} \bar{x_i}^\top P_i X = P_i X
    .. math:: 0 = (P_i - \bar{x_i} \bar{x_i}^\top P_i) X

    We can then stack this constraint for each observation, leading to the linear
    least squares problem:

    .. math:: \begin{bmatrix} (P_1 - \bar{x_1} \bar{x_1}^\top P_1) \\ \vdots \\ (P_n - \bar{x_n} \bar{x_n}^\top P_n) \end{bmatrix} X = \textbf{0}

    This system of equations is of the form :math:`AX=0` which can be solved by
    extracting the right nullspace of :math:`A`. The right nullspace of :math:`A`
    can be extracted efficiently by noting that it is equivalent to the nullspace
    of :math:`A^\top A`, which is a 4x4 matrix.

Similarity Transformation
=========================

  .. cpp:function:: void AlignPointCloudsICP(const int num_points, const double left[], const double right[], double rotation[3 * 3], double translation[3])

    We implement ICP for point clouds. We use Besl-McKay registration to align
    point clouds. We use SVD decomposition to find the rotation, as this is much
    more likely to find the global minimum as compared to traditional ICP, which
    is only guaranteed to find a local minimum. Our goal is to find the
    transformation from the left to the right coordinate system. We assume that
    the left and right models have the same number of points, and that the
    points are aligned by correspondence (i.e. left[i] corresponds to right[i]).

  .. cpp:function:: void AlignPointCloudsUmeyama(const int num_points, const double left[], const double right[], double rotation[3 * 3], double translation[3], double* scale)

    This function estimates the 3D similiarty transformation using the least
    squares method of [Umeyama]_. The returned rotation, translation, and scale
    align the left points to the right such that :math:`Right = s * R * Left +
    t`.

  .. cpp:function:: void DlsSimilarityTransform(const std::vector<Eigen::Vector3d>& ray_origin, const std::vector<Eigen::Vector3d>& ray_direction, const std::vector<Eigen::Vector3d>& world_point, std::vector<Eigen::Quaterniond>* solution_rotation, std::vector<Eigen::Vector3d>* solution_translation, std::vector<double>* solution_scale)

    Computes the solution to the generalized pose and scale problem based on the
    paper "gDLS: A Scalable Solution to the Generalized Pose and Scale Problem"
    by Sweeney et. al. [SweeneyGDLS]_. Given image rays from one coordinate
    system that correspond to 3D points in another coordinate system, this
    function computes the rotation, translation, and scale that will align the
    rays with the 3D points. This is used for applications such as loop closure
    in SLAM and SfM. This method is extremely scalable and highly accurate
    because the cost function that is minimized is independent of the number of
    points. Theoretically, up to 27 solutions may be returned, but in practice
    only 4 real solutions arise and in almost all cases where n >= 6 there is
    only one solution which places the observed points in front of the
    camera. The rotation, translation, and scale are defined such that:
    :math:`sp_i + \alpha_i d_i = RX_i + t` where the observed image ray has an
    origin at :math:`p_i` in the unit direction :math:`d_i` corresponding to 3D
    point :math:`X_i`.

    ``ray_origin``: the origin (i.e., camera center) of the image ray used in
    the 2D-3D correspondence.

    ``ray_direction``: Normalized image rays corresponding to model points. Must
    contain at least 4 points.

    ``world_point``: 3D location of features. Must correspond to the image_ray
    of the same index. Must contain the same number of points as image_ray, and
    at least 4.

    ``solution_rotation``: the rotation quaternion of the candidate solutions

    ``solution_translation``: the translation of the candidate solutions

    ``solution_scale``: the scale of the candidate solutions
