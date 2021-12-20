#include "graphmodule.hpp"

GraphModule::GraphModule()
{
    if(!InitSDL())
        printf("Some problems in inint SDL");

    if(!InitImGui())
        printf("Some problems in inint ImGui");

}

bool GraphModule::InitSDL()
{
    // Setup SDL
    // (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
    // depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return false;
    }

    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);

    const char* glsl_version = "#version 150";

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

    screen = SDL_CreateWindow(
                "SPH",
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                screenWidth,
                screenHeight,
                window_flags);

    if (screen== NULL) {
        std::cerr << "Failed to create main window" << std::endl;
        SDL_Quit();
        return true;
    }


    // Initialize rendering context

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);


    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                       SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );

    gContext = SDL_GL_CreateContext(screen);
    if (gContext == NULL) {
        std::cerr << "Failed to create GL context" << std::endl;
        SDL_DestroyWindow(screen);
        SDL_Quit();
        return true;
    }

    SDL_GL_MakeCurrent(screen, gContext);
    SDL_GL_SetSwapInterval(1);

    //Main loop flag
    bool quit = false;
    SDL_Event e;
    SDL_StartTextInput();

    int major, minor;
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);
    std::cout << "OpenGL version       | " << major << "." << minor << std::endl;
    std::cout << "---------------------+-------" << std::endl;


    return true;
}

bool GraphModule::InitImGui()
{
    // Setup Dear ImGui context
       IMGUI_CHECKVERSION();
       ImGui::CreateContext();
       // Setup Dear ImGui style
       ImGui::StyleColorsDark();

       ImGui::LoadIniSettingsFromDisk("tempImgui.ini");
       ImGuiIO& io = ImGui::GetIO(); (void)io;
       static ImGuiStyle* style = &ImGui::GetStyle();
       style->Alpha = 0.5f;

       io.WantCaptureMouse=true;
       //io.WantCaptureKeyboard=false;
       //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
       //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls


       //ImGui::StyleColorsClassic();

       // Setup Platform/Renderer backends
       ImGui_ImplSDL2_InitForOpenGL(screen, gContext);
       ImGui_ImplOpenGL3_Init("#version 330");

       return true;


}



//OpenGL operations. Because I'm just coloring in points, I don't need to deal with fragment shaders
void GraphModule::Render(Cuda_solver& sph_solver){

    glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewWidth, 0, viewHeight, 0, 1);

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);

   // glEnable(GL_PROGRAM_POINT_SIZE);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    sph_solver.adjustColor();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(drawRatio*particleRadius*sph_solver.h_params->scrRatio);

    glColorPointer(4, GL_FLOAT, sizeof(Vec4), sph_solver.h_savedParticleColors);
    glVertexPointer(2, GL_FLOAT, sizeof(Particle), sph_solver.h_particles);

    glDrawArrays(GL_POINTS, 0, sph_solver.h_params->totalParticles);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}


void GraphModule::Render2(Cuda_solver& sph_solver){
    //glClearColor(0.5f, 0.5f, 0.5f, 1);
    //glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewWidth, 0, viewHeight, 0, 1);

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);


    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   // sph_solver.adjustColor();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(sph_solver.h_params->pixelDiametr*sph_solver.h_params->scrRatio);

    glColorPointer(4, GL_FLOAT, sizeof(Vec4), sph_solver.h_artParticleColors);
    glVertexPointer(2, GL_FLOAT, sizeof(Particle), sph_solver.h_artparticles);
    glDrawArrays(GL_POINTS, 0, sph_solver.h_params->number_artp);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}


void GraphModule::GuiRender(Cuda_solver& sph_solver)
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(screen);
    ImGui::NewFrame();

    {
        ImGui::Begin("SPH particles parameters");
            ImGui::Text("number particles: %i", sph_solver.h_params->totalParticles);
            ImGui::Text("Tick: %i", sph_solver.h_params->tick);
            ImGui::Checkbox("Run simulation", &runSimulation);
            /*
            float  yield=0.08f;
            float  stiffness= 0.18f;
            float  nearStiffness= 0.01f;
            float  linearViscocity =0.5f;
            float  quadraticViscocity= 1.f;
            */

            ImGui::SliderFloat("yield", &sph_solver.h_params->yield, 0.0f, 0.3f);
            ImGui::SliderFloat("stiffness", &sph_solver.h_params->stiffness, 0.0f, 0.5f);
            ImGui::SliderFloat("nearStiffness", &sph_solver.h_params->nearStiffness, 0.0f, 0.15f);
            ImGui::SliderFloat("linearViscocity", &sph_solver.h_params->linearViscocity, 0.0f, 27.5f);
            ImGui::SliderFloat("quadraticViscocity", &sph_solver.h_params->quadraticViscocity, 0.0f, 27.5f);

            sph_solver.h_params->number_artp_ischange=ImGui::SliderInt("Number Points", &sph_solver.h_params->number_artp, 1, 90);
            ImGui::SliderInt("stepDelay", &sph_solver.h_params->stepDelay, 0, 490);
            ImGui::SliderInt("stepDelayRow", &sph_solver.h_params->timeDelayRow, 100, 3500);
            ImGui::SliderInt("stepDelayDigit", &sph_solver.h_params->timeDelayDigit, 100, 3500);
            ImGui::SliderFloat("temp increase", &sph_solver.h_params->temp_increase, 0.5f, 30.0f);
            ImGui::SliderFloat("temp decrease", &sph_solver.h_params->temp_decrease, 0.0f, 0.375f);
            ImGui::SliderFloat("gravity max", &sph_solver.h_params->gravity_coeff_max, 0.0f, 15.0f);
            sph_solver.h_params->gravity.y=-sph_solver.h_params->gravity_coeff_max;
            ImGui::SliderFloat("gravity min", &sph_solver.h_params->gravity_coeff_min, 0.0f, 30.0f);

            ImGui::SliderFloat("magnet radius", &sph_solver.h_params->magnetRadiusCoeff, 0.1f, 3.0f);

            sph_solver.h_params->number_artp_ischange+=ImGui::SliderFloat("Pixel radius", &sph_solver.h_params->pixelRadius, 0.01f, 10.0f);
            sph_solver.h_params->number_artp_ischange+=ImGui::SliderFloat("intraStep radius", &sph_solver.h_params->intraStepPixel, 0.5f, 30.0f);
            if(sph_solver.h_params->number_artp_ischange)
                sph_solver.h_params->pixelDiametr=2.0f*sph_solver.h_params->pixelRadius;
            ImGui::SliderFloat("Radius viscosity", &sph_solver.h_params->radiusViscocity, 0.9f, 1.1f);

            //#define restDensity 75.0f
            //#define surfaceTension 0.0006f
            ImGui::SliderFloat("restDensity", &sph_solver.h_params->restDensity, 10.f, 210.0f);
            ImGui::SliderFloat("surfaceTension", &sph_solver.h_params->surfaceTension, 0.0001f, 0.001f,"%.5f");
            static int item_current = 0;
            const char* items[] = {  "simpleColor",
                                      "tempColor",
                                      "velocityColor" };


            ImGui::Combo("Draw type", &item_current, items, IM_ARRAYSIZE(items));
            sph_solver.h_params->drawListSet=item_current;


            ImGui::Checkbox("Draw ArtPoints", &sph_solver.h_params->drawArtPoint);
            ImGui::SliderFloat("Pixel transperency", &sph_solver.h_params->artPixelTransperency, 0.0f, 1.0f);

            if(ImGui::Button("reset timer"))
            {
                sph_solver.digitTime[2]=0;
                sph_solver.digitTime[3]=0;
            };

        ImGui::End();
    }

    // Rendering
    ImGui::Render();
    /*
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w); 
    glClear(GL_COLOR_BUFFER_BIT);
    */

}

void GraphModule::ClearScreen()
{

}

void GraphModule::CloseRender()
{
    //close program, return true
    SDL_StopTextInput();
    SDL_DestroyWindow(screen);
    screen = NULL;
    SDL_Quit();

}

void  GraphModule::HandleEvents(SDL_Event e, Cuda_solver& sph_solver){
    if (e.type == SDL_KEYDOWN){
        if (e.key.keysym.sym == SDLK_DOWN){
            sph_solver.gravity.x = 0.f;
            sph_solver.gravity.y = -9.8f;
        }
        else if (e.key.keysym.sym == SDLK_UP){
            sph_solver.gravity.x = 0.f;
            sph_solver.gravity.y = 9.8f;
        }
        else if (e.key.keysym.sym == SDLK_LEFT){
            sph_solver.gravity.x = -9.8f;
            sph_solver.gravity.y = 0.f;
        }
        else if (e.key.keysym.sym == SDLK_RIGHT){
            sph_solver.gravity.x = 9.8f;
            sph_solver.gravity.y = 0.f;
        }
        else if (e.key.keysym.sym == SDLK_SPACE){
            sph_solver.gravity.x = 0.f;
            sph_solver.gravity.y = 0.f;
        }
        else if (e.key.keysym.sym == SDLK_LSHIFT){
            if (sph_solver.sources[2].pt == greenParticle){
                printf("1");
                sph_solver.sources[2].pt = blueParticle;
            }
            else if (sph_solver.sources[2].pt == redParticle){
                printf("2");
                sph_solver.sources[2].pt = greenParticle;
            }
        }
        else if (e.key.keysym.sym == SDLK_RSHIFT){
            if (sph_solver.sources[2].pt == greenParticle){
                printf("3");
                sph_solver.sources[2].pt = redParticle;
            }
            else if (sph_solver.sources[2].pt == blueParticle){
                printf("4");
                sph_solver.sources[2].pt = greenParticle;
            }
        }
    }
    /*
    if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT){
        activeSpout = true;
    }
    else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT){
        activeSpout = false;
    }
    */
}
