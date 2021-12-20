#include "mainloop.hpp"

MainLoop::MainLoop()
{

}

bool MainLoop::RunLoop()
{


    bool done = false;
    while (!done)
    {
            // Poll and handle events (inputs, window resize, etc.)
            // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
            // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
            // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
            // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.

            while (SDL_PollEvent(&graphModule.event))
            {
                ImGui_ImplSDL2_ProcessEvent(&graphModule.event);
                if (graphModule.event.type == SDL_QUIT)
                {
                    done = true;
                    ImGui::SaveIniSettingsToDisk("tempImgui.ini");
                }
                if (graphModule.event.type == SDL_WINDOWEVENT && graphModule.event.window.event == SDL_WINDOWEVENT_CLOSE && graphModule.event.window.windowID == SDL_GetWindowID(graphModule.screen))
                    done = true;
                else
                {
                    graphModule.HandleEvents(graphModule.event, cuda_solver);
                }
            }


       // sph_solver.Update();

        if(graphModule.runSimulation)
            cuda_solver.Update();

        /*
        if (activeSpout){
         int x = 0;
         int y = 0;
         SDL_GetMouseState(&x, &y);
         addParticles(x, y);
        }
        update(); // update logic
        */
        graphModule.Render(cuda_solver); // update buffer

       if(cuda_solver.h_params->drawArtPoint)
          graphModule.Render2(cuda_solver);

        graphModule.GuiRender(cuda_solver);
        //glUseProgram(0); // You may want this if using this code in an OpenGL 3+ context where shaders may be bound
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(graphModule.screen); //update window
    }

    graphModule.CloseRender();

    return true;
}
