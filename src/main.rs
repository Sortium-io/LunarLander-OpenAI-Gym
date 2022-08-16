use bevy::{
    app::AppExit,
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    ecs::schedule::ShouldRun,
    input::{keyboard::KeyboardInput, ButtonState},
    prelude::*,
    render::render_resource::PrimitiveTopology,
};
use bevy_rapier3d::{dynamics, prelude::*};
// use crossbeam::channel::bounded;
use simula_camera::orbitcam::*;
use simula_core::{
    prng::Prng,
    signal::{SignalFunction, SignalGenerator},
};
use simula_viz::{
    axes::{Axes, AxesBundle, AxesPlugin},
    grid::{Grid, GridBundle, GridPlugin},
    lines::{LinesMaterial, LinesPlugin},
};
use tokio::runtime::Runtime;

#[derive(Component)]
pub struct TokioRuntime {
    pub runtime: std::sync::Arc<Runtime>,
}

struct SimConfig {
    mode: SimState,
    main_engine_power: f32,
    side_engine_power: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            mode: SimState::Demo,
            main_engine_power: 13.0,
            side_engine_power: 0.6,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum SimState {
    Reset,
    Done,
    // Learn,
    // Play,
    Demo,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum AgentAction {
    Nop,
    LeftEngine,
    MainEngine,
    RightEngine,
}

impl From<AgentAction> for u32 {
    fn from(action: AgentAction) -> Self {
        match action {
            AgentAction::Nop => 0,
            AgentAction::LeftEngine => 1,
            AgentAction::MainEngine => 2,
            AgentAction::RightEngine => 3,
        }
    }
}

impl From<u32> for AgentAction {
    fn from(action: u32) -> Self {
        match action {
            1 => AgentAction::LeftEngine,
            2 => AgentAction::MainEngine,
            3 => AgentAction::RightEngine,
            _ => AgentAction::Nop,
        }
    }
}

#[derive(Default)]
struct SimResources {
    dock_scene: Handle<Scene>,
    rock_scene: Handle<Scene>,
    luna_scene: Handle<Scene>,
    flam_scene: Handle<Scene>,
}

// fn gym_runner(mut app: App) {
//     let runtime = tokio::runtime::Runtime::new().unwrap();

//     runtime.block_on(async move {
//         loop {
//             tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
//             // app.update();
//         }
//     });
// }

fn main() {
    App::new()
        //
        // App and plugins
        .insert_resource(WindowDescriptor {
            title: "[DoubleHELIX] OpenAI Gym - Lunar Lander".to_string(),
            width: 940.,
            height: 528.,
            ..Default::default()
        })
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(AmbientLight {
            brightness: 0.1,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        // .insert_resource(RapierConfiguration {
        //     timestep_mode: TimestepMode::Fixed {
        //         dt: 0.02,
        //         substeps: 1,
        //     },
        //     ..Default::default()
        // })
        .add_plugin(RapierPhysicsPlugin::<NoUserData>::default().with_default_system_setup(false))
        // .add_plugin(RapierDebugRenderPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(OrbitCameraPlugin)
        .add_plugin(LinesPlugin)
        .add_plugin(AxesPlugin)
        .add_plugin(GridPlugin)
        //
        // Simulation setup
        .insert_resource(SimConfig::default())
        .insert_resource(SimResources::default())
        .add_event::<AgentAction>()
        .add_startup_system(setup)
        .add_system(keyboard_event_system)
        .add_system(debug_fps)
        .add_system(debug_state)
        //
        // Custom physics setup
        .add_stage_after(
            CoreStage::Update,
            PhysicsStages::SyncBackend,
            SystemStage::parallel().with_system_set(
                RapierPhysicsPlugin::<NoUserData>::get_systems(PhysicsStages::SyncBackend),
            ),
        )
        .add_stage_after(
            PhysicsStages::SyncBackend,
            PhysicsStages::StepSimulation,
            SystemStage::parallel()
                .with_run_criteria(can_step_simulation)
                .with_system_set(RapierPhysicsPlugin::<NoUserData>::get_systems(
                    PhysicsStages::StepSimulation,
                )),
        )
        .add_stage_after(
            PhysicsStages::StepSimulation,
            PhysicsStages::Writeback,
            SystemStage::parallel().with_system_set(
                RapierPhysicsPlugin::<NoUserData>::get_systems(PhysicsStages::Writeback),
            ),
        )
        .add_stage_before(
            CoreStage::Last,
            PhysicsStages::DetectDespawn,
            SystemStage::parallel().with_system_set(
                RapierPhysicsPlugin::<NoUserData>::get_systems(PhysicsStages::DetectDespawn),
            ),
        )
        //
        // Simulation states
        .add_state(SimState::Demo)
        //
        // State: RESET
        .add_system_set(SystemSet::on_enter(SimState::Reset).with_system(cleanup_sim))
        .add_system_set(SystemSet::on_update(SimState::Reset).with_system(resetting_sim))
        //
        // State: DONE
        .add_system_set(SystemSet::on_update(SimState::Done).with_system(finishing_sim))
        //
        // State: DEMO
        .add_system_set(
            SystemSet::on_enter(SimState::Demo)
                .with_system(cleanup_sim)
                .with_system(setup_terrain)
                .with_system(spawn_agent),
        )
        .add_system_set(
            SystemSet::on_update(SimState::Demo)
                .with_system(kb_control_agent)
                .with_system(run_agent)
                .with_system(run_demo),
        )
        .add_system_set(
            SystemSet::on_exit(SimState::Demo)
                .with_system(stop_agent)
                .with_system(cleanup_demo),
        )
        .add_system_to_stage(CoreStage::PostUpdate, collision_events)
        .run();
}

fn setup(
    mut sim_res: ResMut<SimResources>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut lines_materials: ResMut<Assets<LinesMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Load sim resources
    sim_res.dock_scene = asset_server.load("models/Platform/Platform.gltf#Scene0");
    sim_res.rock_scene = asset_server.load("models/Rock/Rock.gltf#Scene0");
    sim_res.luna_scene = asset_server.load("models/LunarLander/LunarLander.gltf#Scene0");
    sim_res.flam_scene = asset_server.load("models/Flame/Flame.gltf#Scene0");

    // Create a mesh for building lines
    let mut lines_mesh: Mesh = Mesh::new(PrimitiveTopology::LineList);
    lines_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    lines_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
    lines_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
    lines_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new());

    // Grid
    commands
        .spawn_bundle(GridBundle {
            grid: Grid {
                size: 10,
                divisions: 10,
                start_color: Color::BLACK,
                end_color: Color::BLACK,
                ..Default::default()
            },
            mesh: meshes.add(lines_mesh.clone()),
            material: lines_materials.add(LinesMaterial {}),
            transform: Transform::from_translation(Vec3::new(0.0, -0.1, 0.0)),
            ..Default::default()
        })
        .insert(Name::new("Grid"));

    // Axes at world origin
    commands
        .spawn_bundle(AxesBundle {
            axes: Axes {
                size: 5.,
                ..Default::default()
            },
            mesh: meshes.add(lines_mesh.clone()),
            material: lines_materials.add(LinesMaterial {}),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..Default::default()
        })
        .insert(Name::new("Axes: World"));

    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            illuminance: 100000.,
            shadows_enabled: true,
            ..Default::default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 1.0, 1.0, 0.0)),
        ..Default::default()
    });

    // Backdrop
    let mut backdrop_mat = StandardMaterial::default();
    backdrop_mat.unlit = true;
    // backdrop_mat.double_sided = true;
    backdrop_mat.base_color = Color::hsla(0.0, 0.0, 1.0, 1.0);
    backdrop_mat.base_color_texture = Some(asset_server.load("models/Rock/rock_backdrop.jpg"));

    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Capsule {
            radius: 200.0,
            ..Default::default()
        })),
        material: materials.add(backdrop_mat),
        transform: Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::new(1.0, 1.0, -1.0)),
        ..Default::default()
    });

    // Camera
    commands
        .spawn_bundle(Camera3dBundle {
            ..Default::default()
        })
        .insert(OrbitCamera {
            center: Vec3::new(0.0, 16.0, 0.0),
            distance: 80.0,
            ..Default::default()
        });

    // UI text for FPS
    commands
        .spawn_bundle(TextBundle {
            text: Text {
                sections: vec![TextSection {
                    value: "\nFPS: ".to_string(),
                    style: TextStyle {
                        font: asset_server.load("fonts/FiraMono-Medium.ttf"),
                        font_size: 12.0,
                        color: Color::rgb(0.0, 1.0, 0.0),
                    },
                }],
                ..Default::default()
            },
            style: Style {
                position_type: PositionType::Absolute,
                position: UiRect {
                    top: Val::Px(5.0),
                    left: Val::Px(5.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        })
        .insert(TextFPS);

    // UI text for SimState
    commands
        .spawn_bundle(TextBundle {
            text: Text {
                sections: vec![TextSection {
                    value: "\nSTATE: ".to_string(),
                    style: TextStyle {
                        font: asset_server.load("fonts/FiraMono-Medium.ttf"),
                        font_size: 12.0,
                        color: Color::rgb(0.0, 1.0, 0.0),
                    },
                }],
                ..Default::default()
            },
            style: Style {
                position_type: PositionType::Absolute,
                position: UiRect {
                    top: Val::Px(20.0),
                    left: Val::Px(5.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        })
        .insert(TextState);
}

#[derive(Component)]
struct TextFPS;

#[derive(Component)]
struct TextState;

fn debug_fps(diagnostics: Res<Diagnostics>, mut text_fps: Query<&mut Text, With<TextFPS>>) {
    if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(average) = fps.average() {
            for mut text in text_fps.iter_mut() {
                text.sections[0].value = format!("  FPS: {:.2}", average);
            }
        }
    };
}

fn debug_state(state: Res<State<SimState>>, mut text_state: Query<&mut Text, With<TextState>>) {
    for mut text in text_state.iter_mut() {
        text.sections[0].value = format!("STATE: {:?}", state.current());
    }
}

fn keyboard_event_system(
    mut keyboard_input_events: EventReader<KeyboardInput>,
    config: Res<SimConfig>,
    mut state: ResMut<State<SimState>>,
    mut app_exit: EventWriter<AppExit>,
) {
    for event in keyboard_input_events.iter() {
        debug!("{:?}", event);
        match (event.state, event.key_code) {
            (ButtonState::Released, Some(KeyCode::Escape)) => app_exit.send_default(),
            (ButtonState::Released, Some(KeyCode::Space)) => match state.current() {
                SimState::Demo => state.set(SimState::Done).unwrap(),
                SimState::Done => state.set(SimState::Reset).unwrap(),
                SimState::Reset => state.set(config.mode).unwrap(),
            },
            _ => (),
        }
    }
}

fn can_step_simulation(state: Res<State<SimState>>) -> ShouldRun {
    match state.current() {
        SimState::Demo => ShouldRun::Yes,
        _ => ShouldRun::No,
    }
}

fn finishing_sim(config: Res<SimConfig>, mut state: ResMut<State<SimState>>) {
    match config.mode {
        SimState::Demo => state.set(SimState::Demo).unwrap(),
        _ => (),
    }
}

fn resetting_sim(config: Res<SimConfig>, mut state: ResMut<State<SimState>>) {
    match config.mode {
        SimState::Demo => state.set(SimState::Demo).unwrap(),
        _ => (),
    }
}

fn cleanup_sim(mut commands: Commands, sim_objs: Query<Entity, With<SimObject>>) {
    for sim_obj in sim_objs.iter() {
        commands.entity(sim_obj).despawn_recursive();
    }
}

#[derive(Component)]
struct SimObject;

#[derive(Component)]
struct AgentRigidbody;

#[derive(Component)]
struct LeftLeg;

#[derive(Component)]
struct RightLeg;

#[derive(Component, Default)]
struct AgentLeg {
    ground_contact: bool,
}

#[derive(Component)]
struct LeftEngine;

#[derive(Component)]
struct MainEngine;

#[derive(Component)]
struct RightEngine;

#[derive(Component)]
struct AgentEngine {
    intensity: f32,
}

fn spawn_agent(sim_res: Res<SimResources>, mut commands: Commands) {
    let mut rng = Prng::default();

    commands
        .spawn_bundle(SpatialBundle {
            transform: Transform::from_xyz(rng.range_float_range(-40.0, 40.0), 40.0, 0.0),
            ..Default::default()
        })
        .insert(SimObject)
        .with_children(|parent| {
            // Collider: Lander body
            parent
                .spawn_bundle(SpatialBundle {
                    transform: Transform::from_xyz(0.0, 3.5, 0.0),
                    ..Default::default()
                })
                .insert(Collider::cuboid(3.0, 2.0, 3.0))
                .insert(ColliderDebugColor(Color::GREEN))
                .insert(ActiveEvents::COLLISION_EVENTS)
                .insert(ContactForceEventThreshold(0.0));

            // Collider: Lander left leg
            parent
                .spawn_bundle(SpatialBundle {
                    transform: Transform::from_xyz(3.0, 1.0, 0.0),
                    ..Default::default()
                })
                .insert(Collider::cuboid(1.0, 1.0, 1.0))
                .insert(ColliderDebugColor(Color::GREEN))
                .insert(AgentLeg::default())
                .insert(LeftLeg)
                .insert(ActiveEvents::COLLISION_EVENTS)
                .insert(ContactForceEventThreshold(0.0));

            // Collider: Lander right leg
            parent
                .spawn_bundle(SpatialBundle {
                    transform: Transform::from_xyz(-3.0, 1.0, 0.0),
                    ..Default::default()
                })
                .insert(Collider::cuboid(1.0, 1.0, 1.0))
                .insert(ColliderDebugColor(Color::GREEN))
                .insert(AgentLeg::default())
                .insert(RightLeg)
                .insert(ActiveEvents::COLLISION_EVENTS)
                .insert(ContactForceEventThreshold(0.0));

            parent.spawn_bundle(SceneBundle {
                scene: sim_res.luna_scene.clone(),
                transform: Transform::from_rotation(Quat::from_rotation_y(
                    std::f32::consts::FRAC_PI_2,
                )),
                ..default()
            });
            parent
                .spawn_bundle(SceneBundle {
                    scene: sim_res.flam_scene.clone(),
                    transform: Transform::from_xyz(0.0, 0.4, 0.0)
                        .with_rotation(Quat::from_rotation_x(std::f32::consts::PI)),
                    ..default()
                })
                .insert(MainEngine)
                .insert(AgentEngine { intensity: 0.5 });
            parent
                .spawn_bundle(SceneBundle {
                    scene: sim_res.flam_scene.clone(),
                    transform: Transform::from_xyz(2.0, 2.0, 0.0)
                        .with_rotation(Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)),
                    ..default()
                })
                .insert(LeftEngine)
                .insert(AgentEngine { intensity: 0.5 });
            parent
                .spawn_bundle(SceneBundle {
                    scene: sim_res.flam_scene.clone(),
                    transform: Transform::from_xyz(-2.0, 2.0, 0.0)
                        .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
                    ..default()
                })
                .insert(RightEngine)
                .insert(AgentEngine { intensity: 0.5 });
            parent.spawn_bundle(PointLightBundle {
                point_light: PointLight {
                    color: Color::rgb(1.0, 1.0, 1.0),
                    radius: 10.0,
                    intensity: 10000.,
                    ..Default::default()
                },
                transform: Transform::from_xyz(0.0, 2.0, -10.0),
                ..Default::default()
            });
        })
        .insert(AgentRigidbody)
        .insert(RigidBody::Dynamic)
        .insert(dynamics::Velocity::default())
        .insert(dynamics::ExternalImpulse::default())
        .insert(dynamics::Sleeping::default())
        .insert(
            LockedAxes::TRANSLATION_LOCKED_Z
                | LockedAxes::ROTATION_LOCKED_X
                | LockedAxes::ROTATION_LOCKED_Y,
        );
}

fn run_agent(
    config: Res<SimConfig>,
    mut state: ResMut<State<SimState>>,
    time: Res<Time>,
    mut engines: Query<(
        &mut Transform,
        &mut AgentEngine,
        Option<&LeftEngine>,
        Option<&MainEngine>,
        Option<&RightEngine>,
    )>,
    mut actions: EventReader<AgentAction>,
    mut bodies: Query<
        (
            &mut dynamics::ExternalImpulse,
            &dynamics::Sleeping,
            &GlobalTransform,
        ),
        With<AgentRigidbody>,
    >,
) {
    let mut action = AgentAction::Nop;
    for evt_action in actions.iter() {
        action = evt_action.clone();
    }

    // Adjust engine visual effects based on intensity
    for (mut xform, mut engine, left_engine, main_engine, right_engine) in engines.iter_mut() {
        match action {
            AgentAction::LeftEngine if left_engine.is_some() => engine.intensity = 1.0,
            AgentAction::MainEngine if main_engine.is_some() => engine.intensity = 1.0,
            AgentAction::RightEngine if right_engine.is_some() => engine.intensity = 1.0,
            _ => (),
        }
        *xform = xform.with_scale(Vec3::new(
            3.0,
            ((time.time_since_startup().as_secs_f32() * 100.0).sin() * 0.5 + 0.5)
                * 5.0
                * engine.intensity,
            3.0,
        ));

        if engine.intensity > 0.5 {
            engine.intensity -= time.delta_seconds();
        }
    }

    // Apply engine forces based on action
    for (mut force, sleeping, transform) in bodies.iter_mut() {
        match action {
            AgentAction::LeftEngine => {
                force.impulse =
                    transform.mul_vec3(Vec3::new(-config.side_engine_power * 1000.0, 0.0, 0.0))
            }
            AgentAction::MainEngine => {
                force.impulse =
                    transform.mul_vec3(Vec3::new(0.0, config.main_engine_power * 100.0, 0.0))
            }
            AgentAction::RightEngine => {
                force.impulse =
                    transform.mul_vec3(Vec3::new(config.side_engine_power * 1000.0, 0.0, 0.0))
            }
            _ => {
                if sleeping.sleeping {
                    state.set(SimState::Done).unwrap();
                }
            }
        }
    }
}

fn stop_agent(mut thursters: Query<&mut Visibility, With<AgentEngine>>) {
    for mut visibility in thursters.iter_mut() {
        visibility.is_visible = false;
    }
}

fn kb_control_agent(
    _config: Res<SimConfig>,
    mut keyboard_input_events: EventReader<KeyboardInput>,
    mut actions: EventWriter<AgentAction>,
) {
    for event in keyboard_input_events.iter() {
        debug!("{:?}", event);
        match (event.state, event.key_code) {
            (ButtonState::Released, Some(KeyCode::Right)) => actions.send(AgentAction::LeftEngine),
            (ButtonState::Released, Some(KeyCode::Up)) => actions.send(AgentAction::MainEngine),
            (ButtonState::Released, Some(KeyCode::Left)) => actions.send(AgentAction::RightEngine),
            _ => (),
        }
    }
}

#[derive(Component)]
struct LandingDock;

#[derive(Component)]
struct TerrainRock;

fn setup_terrain(mut commands: Commands, sim_res: Res<SimResources>) {
    // Landing dock
    commands
        .spawn_bundle(SpatialBundle::default())
        .insert(SimObject)
        .insert(LandingDock)
        .with_children(|parent| {
            parent.spawn_bundle(SceneBundle {
                scene: sim_res.dock_scene.clone(),
                transform: Transform::from_xyz(0.0, 0.0, 0.0)
                    .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
                ..default()
            });

            parent
                .spawn_bundle(SpatialBundle {
                    transform: Transform::from_xyz(0.0, -1.0, 0.0),
                    ..Default::default()
                })
                .insert(RigidBody::Fixed)
                .insert(Collider::cuboid(5.0, 1.0, 5.0))
                .insert(ColliderDebugColor(Color::BEIGE));
        });

    // Random generator
    let mut rng = Prng::default();

    // Left side rock heights
    let mut west_sig = SignalGenerator {
        func: SignalFunction::GaussNoise,
        amplitude: 20.0,
        frequency: 1.0,
        rng: Prng::default(),
        ..Default::default()
    };

    // Right side rock heights
    let mut east_sig = SignalGenerator {
        func: SignalFunction::GaussNoise,
        amplitude: 20.0,
        frequency: 1.0,
        rng: Prng::default(),
        ..Default::default()
    };

    // Rocks ground
    let west_heights = (0..10).map(move |step| {
        (
            step as f32 + 0.5,
            west_sig.sample(std::time::Duration::from_secs_f32(step as f32 / 10f32)),
        )
    });
    let east_heights = (0..10).map(move |step| {
        (
            -step as f32 - 0.5,
            east_sig.sample(std::time::Duration::from_secs_f32(step as f32 / 10f32)),
        )
    });
    commands
        .spawn_bundle(SpatialBundle::default())
        .insert(SimObject)
        .insert(TerrainRock)
        .with_children(|parent| {
            for (x, y) in west_heights.chain(east_heights) {
                let x = x * 6.0;
                let mut y = y * (x.abs() / 60.).clamp(0., 1.).powf(0.8) - 5.;
                loop {
                    let rot = Quat::from_axis_angle(
                        Vec3::new(rng.rand_float(), rng.rand_float(), rng.rand_float()).normalize(),
                        rng.rand_float() * std::f32::consts::PI,
                    );
                    let mut builder = parent.spawn_bundle(SceneBundle {
                        scene: sim_res.rock_scene.clone(),
                        transform: Transform::from_xyz(x, y, 0.0).with_rotation(rot),
                        ..default()
                    });

                    // Do not add colliders under landing dock
                    if x < -8.0 || x > 8.0 || y < -10.0 {
                        builder
                            .insert(RigidBody::Fixed)
                            .insert(Collider::cuboid(3.0, 3.0, 3.0))
                            .insert(ColliderDebugColor(Color::MAROON));
                    }

                    y -= 6.0;
                    if y < -20.0 {
                        break;
                    }
                }
            }
        });
}

fn cleanup_demo() {}

fn run_demo() {}

fn collision_events(
    mut collision_events: EventReader<CollisionEvent>,
    mut legs: Query<(Entity, &mut AgentLeg)>,
) {
    for collision_event in collision_events.iter() {
        println!("Received collision event: {:?}", collision_event);
        match collision_event {
            CollisionEvent::Started(e0, e1, _) => {
                for (e, mut leg) in legs.iter_mut().find(|(e, _)| e == e0 || e == e1) {
                    info!("Leg collision start: {:?}", e);
                    leg.ground_contact = true;
                }
            }
            CollisionEvent::Stopped(e0, e1, _) => {
                for (e, mut leg) in legs.iter_mut().find(|(e, _)| e == e0 || e == e1) {
                    info!("Leg collision end: {:?}", e);
                    leg.ground_contact = false;
                }
            }
        }
    }
}
